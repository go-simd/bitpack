package bitpack

import (
	"fmt"
	"math/rand"
	"testing"
)

func benchInts(blocks int) []uint32 {
	s := make([]uint32, blocks*BlockSize)
	rng := rand.New(rand.NewSource(1))
	for i := range s {
		s[i] = rng.Uint32()
	}
	return s
}

func packScalarBulk(dst []byte, src []uint32, bits int) {
	for b := 0; b*BlockSize < len(src); b++ {
		packBlockScalar(dst[b*16*bits:], src[b*BlockSize:], bits)
	}
}

func unpackScalarBulk(dst []uint32, src []byte, bits int) {
	for b := 0; b*BlockSize < len(dst); b++ {
		unpackBlockScalar(dst[b*BlockSize:], src[b*16*bits:], bits)
	}
}

func BenchmarkPack(b *testing.B) {
	src := benchInts(128) // 16384 ints
	for _, bits := range []int{8, 16, 24} {
		dst := make([]byte, PackedLen(len(src), bits))
		b.Run(fmt.Sprintf("bits%d/simd", bits), func(b *testing.B) {
			b.SetBytes(int64(len(src) * 4))
			for i := 0; i < b.N; i++ {
				Pack(dst, src, bits)
			}
		})
		b.Run(fmt.Sprintf("bits%d/scalar", bits), func(b *testing.B) {
			b.SetBytes(int64(len(src) * 4))
			for i := 0; i < b.N; i++ {
				packScalarBulk(dst, src, bits)
			}
		})
	}
}

func BenchmarkUnpack(b *testing.B) {
	src := benchInts(128)
	out := make([]uint32, len(src))
	for _, bits := range []int{8, 16, 24} {
		packed := make([]byte, PackedLen(len(src), bits))
		Pack(packed, src, bits)
		b.Run(fmt.Sprintf("bits%d/simd", bits), func(b *testing.B) {
			b.SetBytes(int64(len(src) * 4))
			for i := 0; i < b.N; i++ {
				Unpack(out, packed, bits)
			}
		})
		b.Run(fmt.Sprintf("bits%d/scalar", bits), func(b *testing.B) {
			b.SetBytes(int64(len(src) * 4))
			for i := 0; i < b.N; i++ {
				unpackScalarBulk(out, packed, bits)
			}
		})
	}
}
