//go:build amd64

package bitpack

import (
	"bytes"
	"testing"
)

// packForce / unpackForce drive a chosen kernel (SSE or AVX2) directly over
// whole blocks, so both amd64 paths are exercised even when the runtime CPU (or
// Rosetta) would not dispatch to AVX2. The AVX2 kernel works on block pairs and
// finishes a trailing odd block with SSE.
func packForce(dst []byte, src []uint32, bits int, avx2 bool) {
	blocks := len(src) / BlockSize
	if avx2 && blocks >= 2 {
		pairs := blocks / 2
		packAVX2tab[bits](dst, src, pairs)
		if done := pairs * 2; done < blocks {
			packSSEtab[bits](dst[done*16*bits:], src[done*BlockSize:], 1)
		}
		return
	}
	packSSEtab[bits](dst, src, blocks)
}

func unpackForce(dst []uint32, src []byte, bits int, avx2 bool) {
	blocks := len(dst) / BlockSize
	if avx2 && blocks >= 2 {
		pairs := blocks / 2
		unpackAVX2tab[bits](dst, src, pairs)
		if done := pairs * 2; done < blocks {
			unpackSSEtab[bits](dst[done*BlockSize:], src[done*16*bits:], 1)
		}
		return
	}
	unpackSSEtab[bits](dst, src, blocks)
}

// TestPackForceKernels validates the SSE and AVX2 pack+unpack kernels directly
// against the scalar reference for every width. AVX2 is only exercised when the
// CPU supports it (the instructions would #UD otherwise). The amd64 CI job runs
// on native AVX2 hardware, making it the authoritative gate for the AVX2 path.
func TestPackForceKernels(t *testing.T) {
	for _, avx2 := range []bool{false, true} {
		if avx2 && !hasAVX2 {
			t.Log("AVX2 not available on this CPU; skipping AVX2 force test")
			continue
		}
		for bits := 1; bits <= 32; bits++ {
			// Use an odd block count (5) so the AVX2 path also exercises its
			// trailing-odd-block SSE finisher.
			src := randBlocks(5, bits, int64(bits)*131+int64(b2i(avx2)))
			want := scalarPack(src, bits)

			got := make([]byte, PackedLen(len(src), bits))
			packForce(got, src, bits, avx2)
			if !bytes.Equal(got, want) {
				off := firstDiff(got, want)
				t.Fatalf("pack avx2=%v bits=%d: != scalar at byte %d (got %#x want %#x)",
					avx2, bits, off, got[off], want[off])
			}

			out := make([]uint32, len(src))
			unpackForce(out, got, bits, avx2)
			for i := range src {
				if out[i] != src[i] {
					t.Fatalf("unpack avx2=%v bits=%d i=%d: got %d want %d",
						avx2, bits, i, out[i], src[i])
				}
			}
		}
	}
}

func b2i(b bool) int {
	if b {
		return 1
	}
	return 0
}
