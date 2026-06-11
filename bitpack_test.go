package bitpack

import (
	"bytes"
	"math/rand"
	"testing"
)

// maskTo returns v with only its low `bits` bits kept.
func maskTo(v uint32, bits int) uint32 {
	if bits == 32 {
		return v
	}
	return v & (uint32(1<<uint(bits)) - 1)
}

// randBlocks builds n*128 uint32 values masked to `bits` bits.
func randBlocks(n, bits int, seed int64) []uint32 {
	r := rand.New(rand.NewSource(seed))
	s := make([]uint32, n*BlockSize)
	for i := range s {
		s[i] = maskTo(r.Uint32(), bits)
	}
	return s
}

// scalarPack packs with the pure-Go reference, for byte-exactness checks.
func scalarPack(src []uint32, bits int) []byte {
	dst := make([]byte, PackedLen(len(src), bits))
	for b := 0; b < len(src)/BlockSize; b++ {
		packBlockScalar(dst[b*16*bits:], src[b*BlockSize:], bits)
	}
	return dst
}

// TestRoundTrip packs then unpacks for every width and several block counts.
func TestRoundTrip(t *testing.T) {
	for bits := 1; bits <= 32; bits++ {
		for _, nb := range []int{1, 2, 3, 5, 8} {
			src := randBlocks(nb, bits, int64(bits)*1000+int64(nb))
			dst := make([]byte, PackedLen(len(src), bits))
			n := Pack(dst, src, bits)
			if n != 16*bits*nb {
				t.Fatalf("bits=%d nb=%d: wrote %d want %d", bits, nb, n, 16*bits*nb)
			}
			out := make([]uint32, len(src))
			Unpack(out, dst, bits)
			for i := range src {
				if out[i] != src[i] {
					t.Fatalf("bits=%d nb=%d i=%d: got %d want %d", bits, nb, i, out[i], src[i])
				}
			}
		}
	}
}

// TestPackMatchesScalar checks the active (SIMD on amd64) Pack is byte-identical
// to the scalar reference for every width.
func TestPackMatchesScalar(t *testing.T) {
	for bits := 1; bits <= 32; bits++ {
		src := randBlocks(4, bits, int64(bits)*7+3)
		got := make([]byte, PackedLen(len(src), bits))
		Pack(got, src, bits)
		want := scalarPack(src, bits)
		if !bytes.Equal(got, want) {
			off := firstDiff(got, want)
			t.Fatalf("bits=%d: Pack != scalar at byte %d (got %#x want %#x)",
				bits, off, got[off], want[off])
		}
	}
}

// TestPackMasksHighBits confirms Pack ignores bits above `bits` (matching
// fastpackwithoutmask): packing unmasked data yields the same bytes as packing
// pre-masked data, and unpack recovers the masked values.
func TestPackMasksHighBits(t *testing.T) {
	for bits := 1; bits < 32; bits++ {
		r := rand.New(rand.NewSource(int64(bits) * 99))
		raw := make([]uint32, 2*BlockSize)
		masked := make([]uint32, len(raw))
		for i := range raw {
			raw[i] = r.Uint32()
			masked[i] = maskTo(raw[i], bits)
		}
		a := make([]byte, PackedLen(len(raw), bits))
		b := make([]byte, PackedLen(len(masked), bits))
		Pack(a, raw, bits)
		Pack(b, masked, bits)
		if !bytes.Equal(a, b) {
			t.Fatalf("bits=%d: packing raw vs masked differ", bits)
		}
		out := make([]uint32, len(raw))
		Unpack(out, a, bits)
		for i := range out {
			if out[i] != masked[i] {
				t.Fatalf("bits=%d i=%d: got %d want %d", bits, i, out[i], masked[i])
			}
		}
	}
}

func firstDiff(a, b []byte) int {
	for i := range a {
		if a[i] != b[i] {
			return i
		}
	}
	return len(a)
}

// TestExtremes packs all-zero and all-ones (within bits) blocks.
func TestExtremes(t *testing.T) {
	for bits := 1; bits <= 32; bits++ {
		for _, fill := range []uint32{0, maskTo(0xffffffff, bits)} {
			src := make([]uint32, 3*BlockSize)
			for i := range src {
				src[i] = fill
			}
			dst := make([]byte, PackedLen(len(src), bits))
			Pack(dst, src, bits)
			if want := scalarPack(src, bits); !bytes.Equal(dst, want) {
				t.Fatalf("bits=%d fill=%#x: != scalar", bits, fill)
			}
			out := make([]uint32, len(src))
			Unpack(out, dst, bits)
			for i := range out {
				if out[i] != fill {
					t.Fatalf("bits=%d fill=%#x i=%d got %d", bits, fill, i, out[i])
				}
			}
		}
	}
}

func FuzzPack(f *testing.F) {
	f.Add(uint64(1), 1, 1)
	f.Add(uint64(0xdeadbeef), 24, 3)
	f.Fuzz(func(t *testing.T, seed uint64, bitsRaw, nbRaw int) {
		bits := int(uint(bitsRaw)%32) + 1 // 1..32
		nb := int(uint(nbRaw)%4) + 1      // 1..4 blocks
		src := randBlocks(nb, bits, int64(seed))
		// SIMD output must equal scalar byte-for-byte.
		got := make([]byte, PackedLen(len(src), bits))
		Pack(got, src, bits)
		if want := scalarPack(src, bits); !bytes.Equal(got, want) {
			t.Fatalf("bits=%d nb=%d: Pack != scalar", bits, nb)
		}
		// And round-trip.
		out := make([]uint32, len(src))
		Unpack(out, got, bits)
		for i := range src {
			if out[i] != src[i] {
				t.Fatalf("bits=%d i=%d roundtrip: got %d want %d", bits, i, out[i], src[i])
			}
		}
	})
}

func FuzzUnpack(f *testing.F) {
	f.Add(uint64(7), 13, 2)
	f.Fuzz(func(t *testing.T, seed uint64, bitsRaw, nbRaw int) {
		bits := int(uint(bitsRaw)%32) + 1
		nb := int(uint(nbRaw)%4) + 1
		// Build a valid packed stream from random data, unpack with the active
		// path and with scalar, and require agreement.
		src := randBlocks(nb, bits, int64(seed)+1)
		packed := scalarPack(src, bits)
		gotSIMD := make([]uint32, len(src))
		gotScalar := make([]uint32, len(src))
		Unpack(gotSIMD, packed, bits)
		for b := 0; b < nb; b++ {
			unpackBlockScalar(gotScalar[b*BlockSize:], packed[b*16*bits:], bits)
		}
		for i := range src {
			if gotSIMD[i] != gotScalar[i] {
				t.Fatalf("bits=%d i=%d: SIMD %d != scalar %d", bits, i, gotSIMD[i], gotScalar[i])
			}
		}
	})
}
