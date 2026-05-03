package gateway

import (
	"io"
	"log"

	"github.com/hashicorp/go-hclog"
	"github.com/rs/zerolog"
)

// HCLogAdapter wraps a zerolog.Logger as a hclog.Logger, used by go-plugin's
// internals (handshake, broker, plugin discovery). Identical in spirit to
// the host-side adapter in mass/internal/runtimes/gateway.go.
func HCLogAdapter(logger zerolog.Logger) hclog.Logger {
	return &zlogHCAdapter{logger: logger}
}

type zlogHCAdapter struct {
	logger zerolog.Logger
	name   string
}

func (z *zlogHCAdapter) emit(level hclog.Level, msg string, args ...any) {
	var ev *zerolog.Event
	switch level {
	case hclog.Trace, hclog.Debug:
		ev = z.logger.Debug()
	case hclog.Info, hclog.NoLevel:
		ev = z.logger.Info()
	case hclog.Warn:
		ev = z.logger.Warn()
	case hclog.Error:
		ev = z.logger.Error()
	case hclog.Off:
		return
	}
	for i := 0; i+1 < len(args); i += 2 {
		key, ok := args[i].(string)
		if !ok {
			continue
		}
		ev = ev.Interface(key, args[i+1])
	}
	if z.name != "" {
		ev = ev.Str("hclog_logger", z.name)
	}
	ev.Msg(msg)
}

func (z *zlogHCAdapter) Log(level hclog.Level, msg string, args ...any) { z.emit(level, msg, args...) }
func (z *zlogHCAdapter) Trace(msg string, args ...any)                  { z.emit(hclog.Trace, msg, args...) }
func (z *zlogHCAdapter) Debug(msg string, args ...any)                  { z.emit(hclog.Debug, msg, args...) }
func (z *zlogHCAdapter) Info(msg string, args ...any)                   { z.emit(hclog.Info, msg, args...) }
func (z *zlogHCAdapter) Warn(msg string, args ...any)                   { z.emit(hclog.Warn, msg, args...) }
func (z *zlogHCAdapter) Error(msg string, args ...any)                  { z.emit(hclog.Error, msg, args...) }

func (z *zlogHCAdapter) IsTrace() bool { return zerolog.GlobalLevel() <= zerolog.TraceLevel }
func (z *zlogHCAdapter) IsDebug() bool { return zerolog.GlobalLevel() <= zerolog.DebugLevel }
func (z *zlogHCAdapter) IsInfo() bool  { return zerolog.GlobalLevel() <= zerolog.InfoLevel }
func (z *zlogHCAdapter) IsWarn() bool  { return zerolog.GlobalLevel() <= zerolog.WarnLevel }
func (z *zlogHCAdapter) IsError() bool { return zerolog.GlobalLevel() <= zerolog.ErrorLevel }

func (z *zlogHCAdapter) ImpliedArgs() []any            { return nil }
func (z *zlogHCAdapter) With(args ...any) hclog.Logger { return z }
func (z *zlogHCAdapter) Name() string                  { return z.name }
func (z *zlogHCAdapter) Named(name string) hclog.Logger {
	return &zlogHCAdapter{logger: z.logger.With().Str("plugin_logger", name).Logger(), name: name}
}
func (z *zlogHCAdapter) ResetNamed(name string) hclog.Logger {
	return &zlogHCAdapter{logger: z.logger.With().Str("plugin_logger", name).Logger(), name: name}
}
func (z *zlogHCAdapter) SetLevel(_ hclog.Level) {}
func (z *zlogHCAdapter) GetLevel() hclog.Level  { return hclog.Trace }
func (z *zlogHCAdapter) StandardLogger(_ *hclog.StandardLoggerOptions) *log.Logger {
	return log.New(z, "", 0)
}
func (z *zlogHCAdapter) StandardWriter(_ *hclog.StandardLoggerOptions) io.Writer { return z }
func (z *zlogHCAdapter) Write(p []byte) (int, error) {
	z.logger.Info().Msg(string(p))
	return len(p), nil
}
