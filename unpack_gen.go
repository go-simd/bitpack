//go:build ignore

// Part of the pack_amd64.s generator (run together with pack_gen.go). This file
// emits the unpack kernels unpackBits{bits}_SSE / unpackBits{bits}_AVX2, the
// inverse of the pack kernels. See pack_gen.go for the layout and AVX2
// two-blocks-per-iteration scheme.
//
// Unpack schedule for width `bits`: integer k occupies per-lane bits
// [k*bits, k*bits+bits). With word w=(k*bits)/32 and off=(k*bits)%32, the value
// is (W[w]>>off) | (W[w+1]<<(32-off) when it straddles), masked to `bits`. The
// schedule (which word(s), which shifts) is fixed by `bits`, so each kernel is
// a branch-free unrolled sequence of MOVOU/PSRLD/PSLLD/POR/PAND/MOVOU.
package main

import (
	"fmt"

	"github.com/go-asmgen/asmgen/amd64"
	"github.com/go-asmgen/asmgen/emit"
)

// unpack signature: unpackBitsN(dst []uint32, src []byte, blocks int64).
// DI = dst, SI = src.
func genUnpackSSE(f *emit.File, bits int, maskName string) {
	name := fmt.Sprintf("unpackBits%d_SSE", bits)
	loop, done := "loop_"+name, "done_"+name
	fn := amd64.NewFunc(name, sig(), 0)
	fn.LoadArg("dst_base", "DI").LoadArg("src_base", "SI").LoadArg("blocks", "CX").
		Raw("MOVOU %s+0(SB), X15", maskName).
		Raw("TESTQ CX, CX").Raw("JZ %s", done).
		Label(loop)

	emitUnpackBody(fn, bits, false)

	fn.Raw("ADDQ $%d, SI", 16*bits). // bits words * 16 bytes
		Raw("ADDQ $512, DI").        // 128 ints * 4 bytes
		Raw("DECQ CX").Raw("JNZ %s", loop).
		Label(done).Ret()
	f.Add(fn.Func())
}

func genUnpackAVX2(f *emit.File, bits int, maskName string) {
	name := fmt.Sprintf("unpackBits%d_AVX2", bits)
	loop, done := "loop_"+name, "done_"+name
	fn := amd64.NewFunc(name, sig(), 0)
	fn.LoadArg("dst_base", "DI").LoadArg("src_base", "SI").LoadArg("blocks", "CX").
		Raw("VMOVDQU %s+0(SB), Y15", maskName).
		Raw("TESTQ CX, CX").Raw("JZ %s", done).
		Label(loop)

	emitUnpackBody(fn, bits, true)

	fn.Raw("ADDQ $%d, SI", 32*bits). // two blocks
		Raw("ADDQ $1024, DI").       // two blocks
		Raw("DECQ CX").Raw("JNZ %s", loop).
		Label(done).Raw("VZEROUPPER").Ret()
	f.Add(fn.Func())
}

// emitUnpackBody emits the per-block unpacking. It loads each source word at
// most once, caching the "current" word in Cur (X3/Y3) and loading the next
// word into Nxt (X4/Y4) only when an integer straddles a boundary.
//
//	Cur = X3 / Y3   current source word (already loaded)
//	Nxt = X4 / Y4   next source word (for straddling integers)
//	V   = X1 / Y1   extracted value being assembled
func emitUnpackBody(fn *amd64.Builder, bits int, avx2 bool) {
	cur, nxt, v := "X3", "X4", "X1"
	if avx2 {
		cur, nxt, v = "Y3", "Y4", "Y1"
	}
	avx2BlockBWordBits = bits // for loadWord's block-B source offset
	loaded := -1              // which source word is currently in Cur

	ensure := func(w int) {
		if loaded != w {
			loadWord(fn, cur, w, avx2)
			loaded = w
		}
	}

	for k := 0; k < 32; k++ {
		start := k * bits
		word := start / 32
		off := start % 32
		end := off + bits

		ensure(word)
		// V = Cur >> off
		shiftR(fn, cur, off, v, avx2)
		if end > 32 {
			// straddles: bring in low bits from next word.
			loadWord(fn, nxt, word+1, avx2)
			shiftLInto(fn, nxt, 32-off, v, avx2) // V |= Nxt<<(32-off)
			// Pre-cache next word as current for following integers.
			move2(fn, nxt, cur, avx2)
			loaded = word + 1
		}
		andMask(fn, v, avx2) // V &= mask  (cheap; harmless when bits==32)
		storeVec(fn, v, k, avx2)
	}
}

// loadWord loads source word w. SSE: 16 bytes at SI+16*w. AVX2: low lane = block
// A word w (SI+16*w), high lane = block B word w (SI+16*bits+16*w) -- but bits
// is not known here; we use the stride: block B is at SI+512*? No: in AVX2 the
// two blocks are laid out contiguously, block B starting 16*bits after block A.
// We pass that via the per-call offset computed by the caller. To keep the
// generator simple, AVX2 loads block A from SI+16*w and block B from
// SI+(16*bits)+16*w; bits is captured through the closure in emitUnpackBody.
func loadWord(fn *amd64.Builder, reg string, w int, avx2 bool) {
	if avx2 {
		fn.Raw("VMOVDQU %d(SI), X%s", 16*w, ymmIdx(reg)).
			Raw("VINSERTI128 $1, %d(SI), %s, %s", avx2BlockBWord(w), reg, reg)
		return
	}
	fn.Raw("MOVOU %d(SI), %s", 16*w, reg)
}

// avx2BlockBWordBits holds `bits` for the current unpack kernel so loadWord can
// locate block B's word w. Set by emitUnpackBody via genUnpack* wrappers.
var avx2BlockBWordBits int

func avx2BlockBWord(w int) int { return 16*avx2BlockBWordBits + 16*w }

func shiftR(fn *amd64.Builder, src string, n int, dst string, avx2 bool) {
	if avx2 {
		if n == 0 {
			fn.Raw("VMOVDQA %s, %s", src, dst)
			return
		}
		fn.Raw("VPSRLD $%d, %s, %s", n, src, dst)
		return
	}
	fn.Raw("MOVO %s, %s", src, dst)
	if n != 0 {
		fn.Raw("PSRLL $%d, %s", n, dst)
	}
}

func shiftLInto(fn *amd64.Builder, src string, n int, dst string, avx2 bool) {
	if avx2 {
		fn.Raw("VPSLLD $%d, %s, Y13", n, src).
			Raw("VPOR Y13, %s, %s", dst, dst)
		return
	}
	fn.Raw("MOVO %s, X13", src).
		Raw("PSLLL $%d, X13", n).
		Raw("POR X13, %s", dst)
}

func move2(fn *amd64.Builder, src, dst string, avx2 bool) {
	if avx2 {
		fn.Raw("VMOVDQA %s, %s", src, dst)
		return
	}
	fn.Raw("MOVO %s, %s", src, dst)
}

func andMask(fn *amd64.Builder, v string, avx2 bool) {
	if avx2 {
		fn.Raw("VPAND Y15, %s, %s", v, v)
		return
	}
	fn.Raw("PAND X15, %s", v)
}

func storeVec(fn *amd64.Builder, v string, k int, avx2 bool) {
	if avx2 {
		fn.Raw("VMOVDQU X%s, %d(DI)", ymmIdx(v), 16*k).               // block A
			Raw("VEXTRACTI128 $1, %s, X12", v).
			Raw("VMOVDQU X12, %d(DI)", avx2BlockBWordBits*0+512+16*k) // block B at DI+512
		return
	}
	fn.Raw("MOVOU %s, %d(DI)", v, 16*k)
}
