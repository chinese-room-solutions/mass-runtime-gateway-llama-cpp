// Package download provides a unified file download manager with resumable
// downloads, progress reporting, retries with exponential backoff, and
// optional authentication headers.
package download

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"time"

	"github.com/KernelPryanic/ctxerr"
)

// ManagerInterface abstracts file downloading so callers can swap
// implementations (e.g. for testing).
type ManagerInterface interface {
	// Download fetches url into destPath. Options control resume, progress,
	// retries, and auth headers.
	Download(ctx context.Context, url, destPath string, opts ...Option) error
}

// ProgressFunc is called periodically with bytes downloaded and total bytes.
// total may be -1 if the server did not provide Content-Length.
type ProgressFunc func(downloaded, total int64)

// Manager is the default implementation of ManagerInterface.
type Manager struct {
	client *http.Client
}

// NewManager creates a Manager. If client is nil, http.DefaultClient is used.
func NewManager(client *http.Client) *Manager {
	if client == nil {
		client = http.DefaultClient
	}
	return &Manager{client: client}
}

// options holds per-request configuration.
type options struct {
	headers    http.Header
	progressFn ProgressFunc
	resume     bool
	maxRetries int
	baseDelay  time.Duration
	tempSuffix string // suffix for the temp file (default ".downloading")
}

func defaultOptions() options {
	return options{
		headers:    make(http.Header),
		maxRetries: 3,
		baseDelay:  time.Second,
		tempSuffix: ".downloading",
	}
}

// Option configures a single download request.
type Option func(*options)

// WithHeaders sets extra HTTP headers (e.g. Authorization).
func WithHeaders(h http.Header) Option {
	return func(o *options) {
		for k, vs := range h {
			for _, v := range vs {
				o.headers.Add(k, v)
			}
		}
	}
}

// WithProgress sets a progress callback.
func WithProgress(fn ProgressFunc) Option {
	return func(o *options) { o.progressFn = fn }
}

// WithResume enables resumable download via HTTP Range headers.
// When enabled, a partial temp file is preserved on interruption and
// reused on the next call.
func WithResume(enable bool) Option {
	return func(o *options) { o.resume = enable }
}

// WithMaxRetries sets the maximum number of retry attempts on transient
// errors (connection reset, timeout, 5xx). 0 means no retries.
func WithMaxRetries(n int) Option {
	return func(o *options) { o.maxRetries = n }
}

// WithBaseDelay sets the base delay for exponential backoff between retries.
func WithBaseDelay(d time.Duration) Option {
	return func(o *options) { o.baseDelay = d }
}

// WithTempSuffix overrides the temp file suffix (default ".downloading").
func WithTempSuffix(s string) Option {
	return func(o *options) { o.tempSuffix = s }
}

// sentinel errors for classification.
var (
	ErrHTTPClientError = errors.New("HTTP client error (4xx)")
	ErrHTTPServerError = errors.New("HTTP server error (5xx)")
	ErrWriteFile       = errors.New("writing file")
)

// Download fetches url and saves the result to destPath atomically
// (write to temp, rename on success).
func (m *Manager) Download(ctx context.Context, url, destPath string, opts ...Option) error {
	o := defaultOptions()
	for _, fn := range opts {
		fn(&o)
	}

	dir := filepath.Dir(destPath)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("creating directory %s: %w", dir, err)
	}

	base := filepath.Base(destPath)
	tempPath := filepath.Join(dir, o.tempSuffix+"-"+base)

	// Already downloaded — report full progress and return.
	if info, err := os.Stat(destPath); err == nil && info.Size() > 0 {
		if o.progressFn != nil {
			o.progressFn(info.Size(), info.Size())
		}
		return nil
	}

	var lastErr error
	attempts := 1 + o.maxRetries
	for attempt := range attempts {
		lastErr = m.doDownload(ctx, url, destPath, tempPath, &o)
		if lastErr == nil {
			return nil
		}
		// Don't retry on context cancellation or non-retryable errors.
		if ctx.Err() != nil {
			return lastErr
		}
		if !isRetryable(lastErr) {
			return lastErr
		}
		if attempt < attempts-1 {
			delay := o.baseDelay * (1 << attempt)
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-time.After(delay):
			}
		}
	}
	return lastErr
}

// doDownload performs a single download attempt.
func (m *Manager) doDownload(ctx context.Context, url, destPath, tempPath string, o *options) error {
	var resumeOffset int64
	if o.resume {
		if info, err := os.Stat(tempPath); err == nil {
			resumeOffset = info.Size()
		}
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return fmt.Errorf("creating request: %w", err)
	}
	for k, vs := range o.headers {
		for _, v := range vs {
			req.Header.Add(k, v)
		}
	}
	if resumeOffset > 0 {
		req.Header.Set("Range", fmt.Sprintf("bytes=%d-", resumeOffset))
	}

	resp, err := m.client.Do(req)
	if err != nil {
		return ctxerr.With(fmt.Errorf("HTTP request failed: %w", err), map[string]any{"url": url})
	}
	defer resp.Body.Close() //nolint:errcheck // best-effort cleanup

	resumeOffset, resp, err = m.handleResumeResponse(ctx, resp, url, o, resumeOffset)
	if err != nil {
		return err
	}

	// Compute total size.
	var total int64
	if resp.StatusCode == http.StatusPartialContent {
		total = resumeOffset + resp.ContentLength
	} else {
		total = resp.ContentLength // may be -1
	}

	// Open temp file.
	var f *os.File
	if resumeOffset > 0 && resp.StatusCode == http.StatusPartialContent {
		f, err = os.OpenFile(tempPath, os.O_WRONLY|os.O_APPEND, 0644)
	} else {
		f, err = os.Create(tempPath)
		resumeOffset = 0
	}
	if err != nil {
		return ctxerr.With(fmt.Errorf("opening temp file: %w", err), map[string]any{"path": tempPath})
	}
	defer f.Close() //nolint:errcheck // safety net

	downloaded := resumeOffset
	buf := make([]byte, 64*1024)
	for {
		if ctx.Err() != nil {
			return ctx.Err()
		}

		n, readErr := resp.Body.Read(buf)
		if n > 0 {
			if _, wErr := f.Write(buf[:n]); wErr != nil {
				return fmt.Errorf("%w: %w", ErrWriteFile, wErr) //nolint:errorlint // wrapping sentinel
			}
			downloaded += int64(n)
			if o.progressFn != nil {
				o.progressFn(downloaded, total)
			}
		}
		if readErr == io.EOF {
			break
		}
		if readErr != nil {
			if ctx.Err() != nil {
				return ctx.Err()
			}
			return ctxerr.With(fmt.Errorf("reading response: %w", readErr), map[string]any{"url": url})
		}
	}

	if err := f.Close(); err != nil {
		return fmt.Errorf("closing temp file: %w", err)
	}

	if err := os.Rename(tempPath, destPath); err != nil {
		return ctxerr.With(fmt.Errorf("finalizing download: %w", err), map[string]any{"temp": tempPath, "dest": destPath})
	}

	return nil
}

// handleResumeResponse deals with the various HTTP status codes when resuming.
// Returns the adjusted resumeOffset, the response to read from, and any error.
func (m *Manager) handleResumeResponse(ctx context.Context, resp *http.Response, url string, o *options, resumeOffset int64) (int64, *http.Response, error) {
	switch resp.StatusCode {
	case http.StatusPartialContent:
		return resumeOffset, resp, nil

	case http.StatusOK:
		// Server doesn't support Range or fresh download.
		return 0, resp, nil

	case http.StatusRequestedRangeNotSatisfiable:
		// Partial file invalid — restart fresh.
		// Body close failure here is a network-level issue we can't recover
		// from anyway; the next request will surface the real error.
		if cErr := resp.Body.Close(); cErr != nil {
			return 0, nil, ctxerr.With(fmt.Errorf("closing partial-content response: %w", cErr), map[string]any{"url": url})
		}
		req2, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
		if err != nil {
			return 0, nil, fmt.Errorf("creating request: %w", err)
		}
		for k, vs := range o.headers {
			for _, v := range vs {
				req2.Header.Add(k, v)
			}
		}
		resp2, err := m.client.Do(req2)
		if err != nil {
			return 0, nil, ctxerr.With(fmt.Errorf("HTTP request failed: %w", err), map[string]any{"url": url})
		}
		if resp2.StatusCode != http.StatusOK {
			body, _ := io.ReadAll(io.LimitReader(resp2.Body, 1024))
			if cErr := resp2.Body.Close(); cErr != nil {
				return 0, nil, ctxerr.With(fmt.Errorf("%w: %d: %s (body-close: %v)", httpSentinel(resp2.StatusCode), resp2.StatusCode, string(body), cErr), map[string]any{"url": url, "status": resp2.StatusCode})
			}
			return 0, nil, ctxerr.With(fmt.Errorf("%w: %d: %s", httpSentinel(resp2.StatusCode), resp2.StatusCode, string(body)), map[string]any{"url": url, "status": resp2.StatusCode})
		}
		return 0, resp2, nil

	default:
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 1024))
		return 0, nil, ctxerr.With(fmt.Errorf("%w: %d: %s", httpSentinel(resp.StatusCode), resp.StatusCode, string(body)), map[string]any{"url": url, "status": resp.StatusCode})
	}
}

// TempFilePath returns the temp file path for a given destPath and suffix.
// Useful for cleanup/cancel handlers.
func TempFilePath(destPath, tempSuffix string) string {
	if tempSuffix == "" {
		tempSuffix = ".downloading"
	}
	dir := filepath.Dir(destPath)
	base := filepath.Base(destPath)
	return filepath.Join(dir, tempSuffix+"-"+base)
}

// httpSentinel returns the appropriate sentinel error for the HTTP status code.
func httpSentinel(statusCode int) error {
	if statusCode >= 500 {
		return ErrHTTPServerError
	}
	return ErrHTTPClientError
}

// isRetryable returns true for transient errors that may succeed on retry.
func isRetryable(err error) bool {
	if errors.Is(err, ErrWriteFile) || errors.Is(err, ErrHTTPClientError) {
		return false // disk errors and 4xx are not transient
	}
	// Retry on HTTP 5xx and network errors (connection reset, timeout, etc.).
	// Context cancellations are handled before this is called.
	return true
}
