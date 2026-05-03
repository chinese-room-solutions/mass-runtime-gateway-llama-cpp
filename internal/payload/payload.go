// Package payload encodes/decodes the gateway↔worker job wire format
// (proto: mass-proto/proto/llama-cpp/payload.proto).
//
// The gateway encodes [llamacpp.Job] before calling sched.Schedule;
// MASS forwards the bytes verbatim as HubAssignJob.payload. The worker
// decodes them and routes by job_kind. Streaming results come back the
// same way: each WorkerJobResult.chunk is one encoded [llamacpp.JobChunk].
package payload

import (
	"fmt"

	llamacpp "github.com/chinese-room-solutions/mass-proto/gen/go/llama-cpp"
	"google.golang.org/protobuf/proto"
)

// EncodeJob serialises a Job for transport via HubAssignJob.payload.
func EncodeJob(job *llamacpp.Job) ([]byte, error) {
	b, err := proto.Marshal(job)
	if err != nil {
		return nil, fmt.Errorf("marshalling job: %w", err)
	}
	return b, nil
}

// DecodeJob deserialises bytes back into a Job. Used by the worker.
func DecodeJob(b []byte) (*llamacpp.Job, error) {
	var j llamacpp.Job
	if err := proto.Unmarshal(b, &j); err != nil {
		return nil, fmt.Errorf("unmarshalling job: %w", err)
	}
	return &j, nil
}

// EncodeJobChunk serialises one streaming chunk.
func EncodeJobChunk(c *llamacpp.JobChunk) ([]byte, error) {
	b, err := proto.Marshal(c)
	if err != nil {
		return nil, fmt.Errorf("marshalling chunk: %w", err)
	}
	return b, nil
}

// DecodeJobChunk deserialises one streaming chunk.
func DecodeJobChunk(b []byte) (*llamacpp.JobChunk, error) {
	var c llamacpp.JobChunk
	if err := proto.Unmarshal(b, &c); err != nil {
		return nil, fmt.Errorf("unmarshalling chunk: %w", err)
	}
	return &c, nil
}

// EncodeLoadHints serialises load hints for HubLoadModel.load_hints.
func EncodeLoadHints(h *llamacpp.LoadHints) ([]byte, error) {
	b, err := proto.Marshal(h)
	if err != nil {
		return nil, fmt.Errorf("marshalling load hints: %w", err)
	}
	return b, nil
}

// DecodeLoadHints deserialises bytes back into LoadHints. Used by the worker.
func DecodeLoadHints(b []byte) (*llamacpp.LoadHints, error) {
	var h llamacpp.LoadHints
	if err := proto.Unmarshal(b, &h); err != nil {
		return nil, fmt.Errorf("unmarshalling load hints: %w", err)
	}
	return &h, nil
}

// EncodeLoadedMetadata serialises post-load metadata for
// WorkerLoadModelResult.runtime_metadata. Worker-side helper, included here
// so encode/decode live together.
func EncodeLoadedMetadata(m *llamacpp.LoadedModelMetadata) ([]byte, error) {
	b, err := proto.Marshal(m)
	if err != nil {
		return nil, fmt.Errorf("marshalling loaded metadata: %w", err)
	}
	return b, nil
}

// DecodeLoadedMetadata deserialises post-load metadata. Gateway-side.
func DecodeLoadedMetadata(b []byte) (*llamacpp.LoadedModelMetadata, error) {
	var m llamacpp.LoadedModelMetadata
	if err := proto.Unmarshal(b, &m); err != nil {
		return nil, fmt.Errorf("unmarshalling loaded metadata: %w", err)
	}
	return &m, nil
}
