//go:build ignore

// Command gen produces pack_amd64.s with go-asmgen: SIMD kernels for the
// simdcomp 128-int bit-packing primitive, one specialised kernel per bit width
// 1..32 for both pack and unpack, in SSE2 and AVX2 flavours.
//
// Layout (see bitpack.go): a block of 128 uint32 is 4 interleaved lanes of 32
// integers; lane j carries src[j], src[4+j], ..., src[124+j], packed low-int
// first into the j-th uint32 column of the bits output words. One SSE register
// (4xuint32) is exactly one column-vector, so input vector k = src[4k..4k+3]
// holds the k-th integer of every lane. Packing is therefore a fixed,
// branch-free schedule of PSLLD/POR/PAND with store points wherever a 32-bit
// output word fills, fully determined by `bits` at generation time.
//
// SSE: packBitsN_SSE(dst, src, blocks) processes one 128-int block per
// iteration. AVX2: packBitsN_AVX2 processes TWO blocks per iteration -- a YMM
// holds block A in its low 128-bit lane and block B in its high lane; VPSLLD /
// VPOR / VPAND act per-lane identically, and the two halves are stored to the
// two blocks' output regions. unpack mirrors this.
//
// Run: go run pack_gen.go unpack_gen.go  (or: go run *_gen.go)
package main

import (
	"fmt"
	"os"
	"strings"

	"github.com/go-asmgen/asmgen/abi"
	"github.com/go-asmgen/asmgen/amd64"
	"github.com/go-asmgen/asmgen/emit"
)

func sig() abi.Signature {
	return abi.LayoutArgs(
		[]abi.Arg{abi.Slice("dst"), abi.Slice("src"), abi.Scalar("blocks", abi.Int64)},
		nil,
	)
}

// genPackSSE emits packBits{bits}_SSE: pack one 128-int block per loop.
// Registers: DI=dst, SI=src, CX=blocks. X15 = low-`bits` mask (4x). Acc=X0,
// scratch V=X1.
func genPackSSE(f *emit.File, bits int, maskName string) {
	name := fmt.Sprintf("packBits%d_SSE", bits)
	loop, done := "loop_"+name, "done_"+name
	fn := amd64.NewFunc(name, sig(), 0)
	fn.LoadArg("dst_base", "DI").LoadArg("src_base", "SI").LoadArg("blocks", "CX").
		Raw("MOVOU %s+0(SB), X15", maskName).
		Raw("TESTQ CX, CX").Raw("JZ %s", done).
		Label(loop)

	// Branch-free packing schedule for one block.
	emitPackBody(fn, bits, false)

	fn.Raw("ADDQ $512, SI").                  // 128 ints * 4 bytes
		Raw("ADDQ $%d, DI", 16*bits).         // bits words * 16 bytes
		Raw("DECQ CX").Raw("JNZ %s", loop).
		Label(done).Ret()
	f.Add(fn.Func())
}

// genPackAVX2 emits packBits{bits}_AVX2: pack TWO 128-int blocks per loop.
// DI=dst (block A region), SI=src (block A). Block B is at SI+512 / DI+16*bits.
// A YMM lane-low = block A column, lane-high = block B column.
func genPackAVX2(f *emit.File, bits int, maskName string) {
	name := fmt.Sprintf("packBits%d_AVX2", bits)
	loop, done := "loop_"+name, "done_"+name
	fn := amd64.NewFunc(name, sig(), 0)
	fn.LoadArg("dst_base", "DI").LoadArg("src_base", "SI").LoadArg("blocks", "CX").
		Raw("VMOVDQU %s+0(SB), Y15", maskName). // mask broadcast to both lanes
		Raw("TESTQ CX, CX").Raw("JZ %s", done).
		Label(loop)

	emitPackBody(fn, bits, true)

	fn.Raw("ADDQ $1024, SI").                 // two blocks * 512 bytes
		Raw("ADDQ $%d, DI", 32*bits).         // two blocks * 16*bits bytes
		Raw("DECQ CX").Raw("JNZ %s", loop).
		Label(done).Raw("VZEROUPPER").Ret()
	f.Add(fn.Func())
}

// emitPackBody emits the per-block packing instructions. With AVX2 the same
// schedule runs on YMM (two blocks interleaved low/high lane); the two output
// columns are split with VEXTRACTI128 at each store.
//
// Register convention (SSE / AVX2):
//
//	Acc  = X0 / Y0   running accumulator for the current output word
//	V    = X1 / Y1   freshly loaded (and masked) input vector
//	Ov   = X2 / Y2   overflow carried into the next word
func emitPackBody(fn *amd64.Builder, bits int, avx2 bool) {
	acc, v, ov := "X0", "X1", "X2"
	if avx2 {
		acc, v, ov = "Y0", "Y1", "Y2"
	}
	word := 0       // current output word index
	off := 0        // bit offset within current word
	haveAcc := false // whether Acc holds pending bits for `word`

	for k := 0; k < 32; k++ {
		loadVec(fn, v, k, avx2) // V = masked input vector k

		if off == 0 {
			// Acc starts empty; first contributor of this word.
			move(fn, v, acc, avx2)
			haveAcc = true
		} else {
			shiftLeft(fn, v, off, ov, acc, avx2) // Acc |= V<<off (via tmp Ov)
		}
		end := off + bits
		if end < 32 {
			off = end
			continue
		}
		// Word `word` is now full -> store Acc.
		storeWord(fn, acc, word, bits, avx2)
		word++
		if end > 32 {
			// Overflow bits of this integer start the next word.
			shiftRightInto(fn, v, 32-off, acc, avx2) // Acc = V>>(32-off)
			haveAcc = true
			off = end - 32
		} else {
			haveAcc = false
			off = 0
		}
	}
	if haveAcc {
		// 32*bits is a multiple of 32, so the stream ends on a word boundary;
		// this guards only against generator drift.
		storeWord(fn, acc, word, bits, avx2)
	}
}

// loadVec loads input vector k (src[4k..4k+3], i.e. byte offset 16k) into reg
// and masks it to the low `bits` bits (mask in X15/Y15). For AVX2 the high lane
// loads block B (byte offset 512+16k).
func loadVec(fn *amd64.Builder, reg string, k int, avx2 bool) {
	if avx2 {
		// Y reg: low lane = block A vec k, high lane = block B vec k.
		// VMOVDQU of 16 bytes into XMM then VINSERTI128 the block-B half.
		fn.Raw("VMOVDQU %d(SI), X%s", 16*k, ymmIdx(reg)).
			Raw("VINSERTI128 $1, %d(SI), %s, %s", 512+16*k, reg, reg).
			Raw("VPAND Y15, %s, %s", reg, reg)
		return
	}
	fn.Raw("MOVOU %d(SI), %s", 16*k, reg).
		Raw("PAND X15, %s", reg)
}

func ymmIdx(y string) string { return strings.TrimPrefix(y, "Y") } // "Y1"->"1"

func move(fn *amd64.Builder, src, dst string, avx2 bool) {
	if avx2 {
		fn.Raw("VMOVDQA %s, %s", src, dst)
		return
	}
	fn.Raw("MOVO %s, %s", src, dst)
}

// shiftLeft computes acc |= src<<n using tmp (src is preserved for overflow).
func shiftLeft(fn *amd64.Builder, src string, n int, tmp, acc string, avx2 bool) {
	if avx2 {
		fn.Raw("VPSLLD $%d, %s, %s", n, src, tmp).
			Raw("VPOR %s, %s, %s", tmp, acc, acc)
		return
	}
	fn.Raw("MOVO %s, %s", src, tmp).
		Raw("PSLLL $%d, %s", n, tmp).
		Raw("POR %s, %s", tmp, acc)
}

// shiftRightInto computes acc = src>>n (start a fresh word with overflow bits).
func shiftRightInto(fn *amd64.Builder, src string, n int, acc string, avx2 bool) {
	if avx2 {
		fn.Raw("VPSRLD $%d, %s, %s", n, src, acc)
		return
	}
	fn.Raw("MOVO %s, %s", src, acc).
		Raw("PSRLL $%d, %s", n, acc)
}

// storeWord writes acc to output word `word`. SSE: 16 bytes at DI+16*word.
// AVX2: low lane -> block A word (DI+16*word), high lane -> block B word
// (DI+16*bits+16*word).
func storeWord(fn *amd64.Builder, acc string, word, bits int, avx2 bool) {
	if avx2 {
		fn.Raw("VMOVDQU X%s, %d(DI)", ymmIdx(acc), 16*word).            // block A
			Raw("VEXTRACTI128 $1, %s, X14", acc).
			Raw("VMOVDQU X14, %d(DI)", 16*bits+16*word)                 // block B
		return
	}
	fn.Raw("MOVOU %s, %d(DI)", acc, 16*word)
}

func main() {
	f := emit.NewFile("amd64")

	// One mask constant per width (4x uint32 for SSE, broadcast in AVX2 read of
	// the same 16 bytes... but AVX2 needs 32 bytes; emit a 32-byte version).
	masks := map[int]string{}
	masks32 := map[int]string{}
	for bits := 1; bits <= 32; bits++ {
		m := mask32(bits)
		masks[bits] = f.Data(fmt.Sprintf("mask%d", bits), le4(m))
		masks32[bits] = f.Data(fmt.Sprintf("mask%db", bits), le8(m))
	}

	for bits := 1; bits <= 32; bits++ {
		genPackSSE(f, bits, masks[bits])
		genPackAVX2(f, bits, masks32[bits])
		genUnpackSSE(f, bits, masks[bits])
		genUnpackAVX2(f, bits, masks32[bits])
	}

	if err := os.WriteFile("pack_amd64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote pack_amd64.s")

	if err := os.WriteFile("pack_decl_amd64.go", []byte(genDecls()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote pack_decl_amd64.go")
}

// genDecls emits the Go-side prototypes for every generated kernel plus the
// dispatch tables packSSEtab/packAVX2tab/unpackSSEtab/unpackAVX2tab indexed by
// bit width. Slice element types differ from the assembly's generic slice view
// (header layout is identical), so dst/src are typed []byte vs []uint32 to suit
// each direction's call site.
func genDecls() string {
	var b strings.Builder
	b.WriteString("// Code generated by go-asmgen (pack_gen.go). DO NOT EDIT.\n\n")
	b.WriteString("//go:build amd64\n\npackage bitpack\n\n")
	for bits := 1; bits <= 32; bits++ {
		fmt.Fprintf(&b, "func packBits%d_SSE(dst []byte, src []uint32, blocks int)\n", bits)
		fmt.Fprintf(&b, "func packBits%d_AVX2(dst []byte, src []uint32, blocks int)\n", bits)
		fmt.Fprintf(&b, "func unpackBits%d_SSE(dst []uint32, src []byte, blocks int)\n", bits)
		fmt.Fprintf(&b, "func unpackBits%d_AVX2(dst []uint32, src []byte, blocks int)\n", bits)
	}
	b.WriteString("\nvar packSSEtab = [33]func(dst []byte, src []uint32, blocks int){\n")
	for bits := 1; bits <= 32; bits++ {
		fmt.Fprintf(&b, "\t%d: packBits%d_SSE,\n", bits, bits)
	}
	b.WriteString("}\n\nvar packAVX2tab = [33]func(dst []byte, src []uint32, blocks int){\n")
	for bits := 1; bits <= 32; bits++ {
		fmt.Fprintf(&b, "\t%d: packBits%d_AVX2,\n", bits, bits)
	}
	b.WriteString("}\n\nvar unpackSSEtab = [33]func(dst []uint32, src []byte, blocks int){\n")
	for bits := 1; bits <= 32; bits++ {
		fmt.Fprintf(&b, "\t%d: unpackBits%d_SSE,\n", bits, bits)
	}
	b.WriteString("}\n\nvar unpackAVX2tab = [33]func(dst []uint32, src []byte, blocks int){\n")
	for bits := 1; bits <= 32; bits++ {
		fmt.Fprintf(&b, "\t%d: unpackBits%d_AVX2,\n", bits, bits)
	}
	b.WriteString("}\n")
	return b.String()
}

func mask32(bits int) uint32 {
	if bits == 32 {
		return 0xffffffff
	}
	return uint32(1<<uint(bits)) - 1
}

func le4(v uint32) []byte {
	b := make([]byte, 16)
	for i := 0; i < 4; i++ {
		b[i*4+0] = byte(v)
		b[i*4+1] = byte(v >> 8)
		b[i*4+2] = byte(v >> 16)
		b[i*4+3] = byte(v >> 24)
	}
	return b
}

func le8(v uint32) []byte {
	b := make([]byte, 32)
	for i := 0; i < 8; i++ {
		b[i*4+0] = byte(v)
		b[i*4+1] = byte(v >> 8)
		b[i*4+2] = byte(v >> 16)
		b[i*4+3] = byte(v >> 24)
	}
	return b
}
