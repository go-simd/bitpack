//go:build amd64

package bitpack

import (
	"bytes"
	"testing"
)

// TestPackBlock exercises the amd64 single-block packBlock/unpackBlock helpers
// (thin wrappers over the per-width SSE kernel) for every width, checking the
// packed bytes against the scalar reference and that unpackBlock recovers the
// values. These helpers run the SSE2 kernel, which is baseline on every amd64.
func TestPackBlock(t *testing.T) {
	for bits := 1; bits <= 32; bits++ {
		src := randBlocks(1, bits, int64(bits)*53+9)
		got := make([]byte, PackedLen(len(src), bits))
		packBlock(got, src, bits)
		if want := scalarPack(src, bits); !bytes.Equal(got, want) {
			off := firstDiff(got, want)
			t.Fatalf("bits=%d: packBlock != scalar at byte %d (got %#x want %#x)",
				bits, off, got[off], want[off])
		}
		out := make([]uint32, len(src))
		unpackBlock(out, got, bits)
		for i := range src {
			if out[i] != src[i] {
				t.Fatalf("bits=%d i=%d: unpackBlock got %d want %d", bits, i, out[i], src[i])
			}
		}
	}
}
