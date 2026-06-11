//go:build ignore

// Command gens390x produces pack_s390x.s with go-asmgen: vector-facility kernels
// for the simdcomp 128-int bit-packing primitive, one specialised kernel per bit
// width 1..32 for both pack and unpack.
//
// Layout (see bitpack.go): a block of 128 uint32 is 4 interleaved lanes of 32
// integers; lane j carries src[j], src[4+j], ..., src[124+j], packed low-int
// first into the j-th uint32 column of the bits output words. One vector
// register (4xuint32) is one column-vector, so input vector k = src[4k..4k+3]
// holds the k-th integer of every lane. Packing is a fixed, branch-free schedule
// of VESLF/VO/VN with store points wherever a 32-bit output word fills, fully
// determined by `bits` at generation time. unpack mirrors it.
//
// # BIG-ENDIAN byte order (the crux of this port)
//
// s390x is big-endian, so the data has two distinct byte orders:
//
//   - The Go uint32 arrays (src on pack, dst on unpack) are stored NATIVE, i.e.
//     big-endian: uint32(1) is the memory bytes 00 00 00 01. The vector facility
//     also numbers bytes big-endian (byte 0 = most-significant), so a plain VL of
//     a uint32 array places the numerically-correct value into each word element
//     with NO byte reversal -- and a plain VST of register-correct words writes
//     them back correctly.
//   - The PACKED bitstream must be little-endian, byte-identical to Lemire's
//     simdcomp (and to the amd64/scalar output). That is the only place a byte
//     order mismatch arises.
//
// The Go s390x assembler exposes no VLBRF/VSTBRF byte-reversing load/store, so we
// byte-reverse each 32-bit word in-register with VPERM and a constant selector
// (selrev = [3,2,1,0, 7,6,5,4, 11,10,9,8, 15,14,13,12]) only when touching the
// packed stream: just before storing a packed word on pack (zStoreWord) and just
// after loading a packed word on unpack (zLoadWord). Loads of src and stores of
// dst use a plain VL/VST. Element shifts (VESLF/VESRLF) then always operate on
// register-correct magnitudes. The per-width mask is stored register-correct
// (big-endian word bytes) so it needs no reversal.
//
// Run: go run pack_s390x_gen.go unpack_s390x_gen.go
package main

import (
	"fmt"
	"os"
	"strings"

	"github.com/go-asmgen/asmgen/abi"
	"github.com/go-asmgen/asmgen/emit"
	"github.com/go-asmgen/asmgen/s390x"
)

func zSig() abi.Signature {
	return abi.LayoutArgs(
		[]abi.Arg{abi.Slice("dst"), abi.Slice("src"), abi.Scalar("blocks", abi.Int64)},
		nil,
	)
}

// zState holds the shared selector/mask symbols for the s390x file.
type zState struct {
	f      *emit.File
	selSym string
}

// genPackZ emits packBits{bits}_VX: pack one 128-int block per loop.
// R2=dst, R3=src, R4=blocks. V10=selrev, V11=mask. Acc=V0, V=V1, Ov=V2.
func (s *zState) genPackZ(bits int, maskSym string) {
	name := fmt.Sprintf("packBits%d_VX", bits)
	loop, done := "loop_"+name, "done_"+name
	fn := s390x.NewFunc(name, zSig(), 0)
	fn.LoadArg("dst_base", "R2").LoadArg("src_base", "R3").LoadArg("blocks", "R4").
		Raw("MOVD $%s(SB), R5", s.selSym).
		Raw("VL (R5), V10"). // selrev
		Raw("MOVD $%s(SB), R5", maskSym).
		Raw("VL (R5), V11"). // mask (already register-correct)
		Raw("CMPBEQ R4, $0, %s", done).
		Label(loop)

	s.emitPackBodyZ(fn, bits)

	fn.Raw("ADD $512, R3").       // 128 ints * 4 bytes
		Raw("ADD $%d, R2", 16*bits). // bits words * 16 bytes
		Raw("ADD $-1, R4").Raw("CMPBNE R4, $0, %s", loop).
		Label(done).Ret()
	s.f.Add(fn.Func())
}

// emitPackBodyZ mirrors the amd64 SSE emitPackBody schedule using the vector
// facility, with VPERM byte-reversal at load and store.
//
//	Acc = V0   running accumulator for the current output word
//	V   = V1   freshly loaded (reversed + masked) input vector
//	Ov  = V2   overflow scratch
func (s *zState) emitPackBodyZ(fn *s390x.Builder, bits int) {
	const acc, v, ov = "V0", "V1", "V2"
	word := 0
	off := 0
	haveAcc := false

	for k := 0; k < 32; k++ {
		zLoadVec(fn, v, k)               // V = bswap(src[k]) (register-correct)
		fn.Raw("VN V11, %s, %s", v, v)   // V &= mask

		if off == 0 {
			fn.Raw("VLR %s, %s", v, acc) // acc = v
			haveAcc = true
		} else {
			// acc |= v << off
			fn.Raw("VESLF $%d, %s, %s", off, v, ov). // Ov = v << off
				Raw("VO %s, %s, %s", ov, acc, acc)
		}
		end := off + bits
		if end < 32 {
			off = end
			continue
		}
		zStoreWord(fn, acc, word) // store (reverses to LE)
		word++
		if end > 32 {
			fn.Raw("VESRLF $%d, %s, %s", 32-off, v, acc) // acc = v >> (32-off)
			haveAcc = true
			off = end - 32
		} else {
			haveAcc = false
			off = 0
		}
	}
	if haveAcc {
		zStoreWord(fn, acc, word)
	}
}

// zLoadVec loads src vector k (byte offset 16k) into reg. The src []uint32 array
// is stored in native (big-endian) order on s390x, so a plain VL already places
// the numerically-correct uint32 into each word element -- no byte reversal is
// needed here. Reversal happens only when touching the little-endian PACKED
// stream (zStoreWord on pack, zLoadWord on unpack).
func zLoadVec(fn *s390x.Builder, reg string, k int) {
	if k == 0 {
		fn.Raw("VL (R3), %s", reg)
		return
	}
	fn.Raw("MOVD $%d, R6", 16*k).
		Raw("VL (R3)(R6*1), %s", reg)
}

// zStoreWord byte-reverses reg back to little-endian and stores to output word
// `word` (byte offset 16*word).
func zStoreWord(fn *s390x.Builder, reg string, word int) {
	fn.Raw("VPERM %s, %s, V10, V5", reg, reg) // V5 = LE bytes of reg
	if word == 0 {
		fn.Raw("VST V5, (R2)")
		return
	}
	fn.Raw("MOVD $%d, R6", 16*word).
		Raw("VST V5, (R2)(R6*1)")
}

func zMain() {
	f := emit.NewFile("s390x")
	s := &zState{f: f}
	// Byte-reverse-per-word selector, register-correct constant: bytes
	// [3,2,1,0, 7,6,5,4, 11,10,9,8, 15,14,13,12].
	s.selSym = f.Data("selrev", []byte{3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12})

	masks := map[int]string{}
	for bits := 1; bits <= 32; bits++ {
		masks[bits] = f.Data(fmt.Sprintf("zmask%d", bits), beSplat4(mask32z(bits)))
	}
	for bits := 1; bits <= 32; bits++ {
		s.genPackZ(bits, masks[bits])
		s.genUnpackZ(bits, masks[bits])
	}

	if err := os.WriteFile("pack_s390x.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote pack_s390x.s")

	if err := os.WriteFile("pack_decl_s390x.go", []byte(genDeclsZ()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote pack_decl_s390x.go")
}

func genDeclsZ() string {
	var b strings.Builder
	b.WriteString("// Code generated by go-asmgen (pack_s390x_gen.go). DO NOT EDIT.\n\n")
	b.WriteString("//go:build s390x\n\npackage bitpack\n\n")
	for bits := 1; bits <= 32; bits++ {
		fmt.Fprintf(&b, "func packBits%d_VX(dst []byte, src []uint32, blocks int)\n", bits)
		fmt.Fprintf(&b, "func unpackBits%d_VX(dst []uint32, src []byte, blocks int)\n", bits)
	}
	b.WriteString("\nvar packVXtab = [33]func(dst []byte, src []uint32, blocks int){\n")
	for bits := 1; bits <= 32; bits++ {
		fmt.Fprintf(&b, "\t%d: packBits%d_VX,\n", bits, bits)
	}
	b.WriteString("}\n\nvar unpackVXtab = [33]func(dst []uint32, src []byte, blocks int){\n")
	for bits := 1; bits <= 32; bits++ {
		fmt.Fprintf(&b, "\t%d: unpackBits%d_VX,\n", bits, bits)
	}
	b.WriteString("}\n")
	return b.String()
}

func mask32z(bits int) uint32 {
	if bits == 32 {
		return 0xffffffff
	}
	return uint32(1<<uint(bits)) - 1
}

// beSplat4 returns v replicated into 4 big-endian uint32 words (16 bytes), the
// in-memory form a plain VL reads as the register-correct value [v,v,v,v].
func beSplat4(v uint32) []byte {
	b := make([]byte, 16)
	for i := 0; i < 4; i++ {
		b[i*4+0] = byte(v >> 24)
		b[i*4+1] = byte(v >> 16)
		b[i*4+2] = byte(v >> 8)
		b[i*4+3] = byte(v)
	}
	return b
}

func main() { zMain() }
