package bitpack

import "encoding/binary"

func bitMask(bits int) uint32 {
	if bits == 32 {
		return 0xffffffff
	}
	return uint32(1<<uint(bits)) - 1
}

// packBlockScalar packs one 128-int block from src into dst[:16*bits] using the
// 4-lane interleaved simdcomp layout. It is the portable reference: the amd64
// SIMD kernels must produce byte-identical output. Each lane holds 32 integers,
// and 32*bits is always a multiple of 32, so every lane's bitstream ends
// exactly on an output-word boundary.
func packBlockScalar(dst []byte, src []uint32, bits int) {
	mask := bitMask(bits)
	var words [32][4]uint32 // output words 0..bits-1, 4 lanes each
	for lane := 0; lane < 4; lane++ {
		bitpos, word := 0, 0
		var acc uint32
		for k := 0; k < 32; k++ {
			v := src[k*4+lane] & mask
			acc |= v << uint(bitpos)
			next := bitpos + bits
			if next < 32 {
				bitpos = next
				continue
			}
			words[word][lane] = acc
			word++
			if next > 32 {
				acc = v >> uint(32-bitpos)
			} else {
				acc = 0
			}
			bitpos = next - 32
		}
	}
	for w := 0; w < bits; w++ {
		base := w * 16
		binary.LittleEndian.PutUint32(dst[base+0:], words[w][0])
		binary.LittleEndian.PutUint32(dst[base+4:], words[w][1])
		binary.LittleEndian.PutUint32(dst[base+8:], words[w][2])
		binary.LittleEndian.PutUint32(dst[base+12:], words[w][3])
	}
}

// unpackBlockScalar is the inverse of packBlockScalar.
func unpackBlockScalar(dst []uint32, src []byte, bits int) {
	mask := bitMask(bits)
	var words [32][4]uint32
	for w := 0; w < bits; w++ {
		base := w * 16
		words[w][0] = binary.LittleEndian.Uint32(src[base+0:])
		words[w][1] = binary.LittleEndian.Uint32(src[base+4:])
		words[w][2] = binary.LittleEndian.Uint32(src[base+8:])
		words[w][3] = binary.LittleEndian.Uint32(src[base+12:])
	}
	for lane := 0; lane < 4; lane++ {
		bitpos, word := 0, 0
		for k := 0; k < 32; k++ {
			v := words[word][lane] >> uint(bitpos)
			next := bitpos + bits
			switch {
			case next < 32:
				bitpos = next
			case next == 32:
				word++
				bitpos = 0
			default:
				word++
				v |= words[word][lane] << uint(32-bitpos)
				bitpos = next - 32
			}
			dst[k*4+lane] = v & mask
		}
	}
}
