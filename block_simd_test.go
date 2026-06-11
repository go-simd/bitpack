//go:build amd64 || ppc64le || s390x

package bitpack

import (
	"bytes"
	"testing"
)

// TestPackBlock exercises the single-block packBlock/unpackBlock helpers (thin
// wrappers over the per-width SIMD kernel) for every width, checking the packed
// bytes against the scalar reference and that unpackBlock recovers the values.
// These helpers run the baseline SIMD kernel of each arch (SSE2 on amd64, VSX on
// ppc64le, the vector facility on s390x).
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
