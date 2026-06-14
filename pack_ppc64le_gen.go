//go:build ignore

// Command genppc64le produces pack_ppc64le.s with go-asmgen: VSX kernels for the
// simdcomp 128-int bit-packing primitive, one specialised kernel per bit width
// 1..32 for both pack and unpack.
//
// Layout (see bitpack.go): a block of 128 uint32 is 4 interleaved lanes of 32
// integers; lane j carries src[j], src[4+j], ..., src[124+j], packed low-int
// first into the j-th uint32 column of the bits output words. One VSX register
// (4xuint32) is exactly one column-vector, so input vector k = src[4k..4k+3]
// holds the k-th integer of every lane. Packing is therefore a fixed,
// branch-free schedule of VSLW/VOR/VAND with store points wherever a 32-bit
// output word fills, fully determined by `bits` at generation time. unpack
// mirrors it.
//
// # Endianness / VSX-VMX aliasing
//
// ppc64le is little-endian. The packed bitstream and the src []uint32 array are
// both little-endian, so word-element loads/stores must preserve LE word order:
// we use LXVW4X / STXVW4X, which place memory word i into register word element
// i (verified under qemu). Plain LXVD2X would byte/doubleword-permute and is NOT
// used.
//
// The VSX register file aliases the VMX (AltiVec) file as Vn == VS(32+n).
// LXVW4X/STXVW4X name VSX registers (VS32..VS63 for the V0..V31 we compute on),
// while the arithmetic (VSLW/VSRW/VAND/VOR) names VMX registers (V0..V31). The
// generator keeps a strict mapping: data/scratch live in V0..V13 and are loaded
// as VS32..VS45; the per-width mask is V15==VS47; shift-count constants are
// loaded on demand into V14==VS46.
//
// Shift amounts are compile-time constants but VSLW/VSRW take a per-word vector
// count, so each needed count 1..31 is materialised as a 4xuint32 data constant
// and loaded with LXVW4X. (VSPLTISW only reaches -16..15 and is not used.)
//
// Run: go run pack_ppc64le_gen.go unpack_ppc64le_gen.go
package main

import (
	"fmt"
	"os"
	"sort"
	"strings"

	"github.com/go-asmgen/asmgen/abi"
	"github.com/go-asmgen/asmgen/emit"
	"github.com/go-asmgen/asmgen/ppc64"
)

func ppcSig() abi.Signature {
	return abi.LayoutArgs(
		[]abi.Arg{abi.Slice("dst"), abi.Slice("src"), abi.Scalar("blocks", abi.Int64)},
		nil,
	)
}

// vsOf returns the VSX register name (VS(32+n)) for VMX register "Vn", used by
// LXVW4X/STXVW4X. e.g. vsOf("V0") == "VS32".
func vsOf(v string) string {
	var n int
	fmt.Sscanf(v, "V%d", &n)
	return fmt.Sprintf("VS%d", 32+n)
}

// ppcState carries the per-kernel constant pool: a set of shift counts that the
// body needs, each emitted once as a data symbol.
//
// The shift counts are loop-invariant (they are compile-time constants, just
// materialised as 4xuint32 vectors because VSLW/VSRW take a per-word count
// vector). The earlier generator reloaded each one with MOVD+LXVW4X into the
// single scratch register V14 at every use, i.e. on every loop iteration. That
// is pure waste: the value never changes. We now HOIST them — each distinct
// count a kernel needs is loaded ONCE into its own dedicated VMX register before
// Label("loop"), and the body just names that register.
//
// Register budget (VMX is V0..V31; the arithmetic VSLW/VSRW are VMX-only, so
// counts must live in V0..V31, not the wider VS0..VS63):
//
//	pack   fixed: V0=acc, V1=v, V2=ov, V15=mask
//	unpack fixed: V1=v, V3=cur, V4=nxt, V13=scratch, V15=mask
//	V14   = fallback on-demand reload register (only used when the pool spills)
//
// hoistPool is the set of registers free in BOTH kernels, used to hold hoisted
// counts. It has 24 slots; widths needing <=24 distinct counts (every width
// except the odd/coprime ones, which need up to 31) hoist all of them, and the
// few remaining widths hoist 24 and reload the spillover from V14 (still
// correct, just not fully hoisted). countReg maps a count to its hoisted
// register for the current kernel; counts absent from it spill to V14.
var hoistPool = func() []string {
	var p []string
	for n := 5; n <= 12; n++ {
		p = append(p, fmt.Sprintf("V%d", n))
	}
	for n := 16; n <= 31; n++ {
		p = append(p, fmt.Sprintf("V%d", n))
	}
	return p
}()

type ppcState struct {
	f        *emit.File
	countSym map[int]string // shift amount -> data symbol (deduped data pool)
	maskSym  string
	maskBits int
	countReg map[int]string // shift amount -> hoisted VMX register (per kernel)
}

// count returns the data symbol for shift amount n, emitting it once.
func (s *ppcState) count(n int) string {
	if sym, ok := s.countSym[n]; ok {
		return sym
	}
	sym := s.f.Data(fmt.Sprintf("cnt%d", n), splat4(uint32(n)))
	s.countSym[n] = sym
	return sym
}

// hoistCounts assigns each distinct shift count in `need` (sorted ascending) a
// dedicated register from hoistPool and emits the one-time LXVW4X load before the
// loop body. Counts beyond the pool's capacity are left unassigned and fall back
// to the V14 on-demand reload at their use sites. It (re)initialises s.countReg
// for the current kernel.
func (s *ppcState) hoistCounts(fn *ppc64.Builder, need []int) {
	s.countReg = map[int]string{}
	for i, n := range need {
		if i >= len(hoistPool) {
			break // pool exhausted; remaining counts reload from V14
		}
		reg := hoistPool[i]
		s.countReg[n] = reg
		fn.Raw("MOVD $%s(SB), R7", s.count(n)).
			Raw("LXVW4X (R7), %s", vsOf(reg)) // reg = splat(n), loaded once
	}
}

// countVecReg returns the register holding the shift-count vector for n. If n
// was hoisted, that is its dedicated register; otherwise it falls back to a
// fresh MOVD+LXVW4X reload into the scratch V14 and returns "V14".
func (s *ppcState) countVecReg(fn *ppc64.Builder, n int) string {
	if reg, ok := s.countReg[n]; ok {
		return reg
	}
	fn.Raw("MOVD $%s(SB), R7", s.count(n)).
		Raw("LXVW4X (R7), %s", vsOf("V14"))
	return "V14"
}

// packCounts returns the distinct shift counts (sorted ascending) the pack body
// for width `bits` will use — mirrors emitPackBodyPPC's schedule exactly so the
// hoist pass and the emit pass agree.
func packCounts(bits int) []int {
	set := map[int]bool{}
	off := 0
	for k := 0; k < 32; k++ {
		if off != 0 {
			set[off] = true // shiftLeftOrPPC n=off
		}
		end := off + bits
		if end < 32 {
			off = end
			continue
		}
		if end > 32 {
			set[32-off] = true // shiftRightIntoPPC n=32-off
			off = end - 32
		} else {
			off = 0
		}
	}
	return sortedKeys(set)
}

// unpackCounts mirrors emitUnpackBodyPPC's schedule.
func unpackCounts(bits int) []int {
	set := map[int]bool{}
	for k := 0; k < 32; k++ {
		off := (k * bits) % 32
		end := off + bits
		if off != 0 {
			set[off] = true // VSRW off
		}
		if end > 32 {
			set[32-off] = true // VSLW 32-off
		}
	}
	return sortedKeys(set)
}

func sortedKeys(set map[int]bool) []int {
	out := make([]int, 0, len(set))
	for n := range set {
		out = append(out, n)
	}
	sort.Ints(out)
	return out
}

// genPackPPC emits packBits{bits}_VSX: pack one 128-int block per loop.
// R3=dst, R4=src, R5=blocks. V15=mask (loaded once). Acc=V0, V=V1, Ov=V2,
// count scratch=V14. The loop-invariant shift-count vectors are hoisted into
// dedicated registers before Label("loop") (see hoistCounts).
func genPackPPC(s *ppcState, bits int, maskSym string) {
	name := fmt.Sprintf("packBits%d_VSX", bits)
	loop, done := "loop_"+name, "done_"+name
	s.maskSym, s.maskBits = maskSym, bits
	fn := ppc64.NewFunc(name, ppcSig(), 0)
	fn.LoadArg("dst_base", "R3").LoadArg("src_base", "R4").LoadArg("blocks", "R5").
		Raw("MOVD $%s(SB), R6", maskSym).
		Raw("LXVW4X (R6), %s", vsOf("V15")). // V15 = mask
		Raw("CMP R5, $0").Raw("BEQ %s", done)
	s.hoistCounts(fn, packCounts(bits)) // load invariant shift counts once
	fn.Label(loop)

	s.emitPackBodyPPC(fn, bits)

	fn.Raw("ADD $512, R4").       // 128 ints * 4 bytes
		Raw("ADD $%d, R3", 16*bits). // bits words * 16 bytes
		Raw("ADD $-1, R5").Raw("CMP R5, $0").Raw("BNE %s", loop).
		Label(done).Ret()
	s.f.Add(fn.Func())
}

// emitPackBodyPPC mirrors the amd64 SSE emitPackBody schedule using VSX.
//
//	Acc = V0   running accumulator for the current output word
//	V   = V1   freshly loaded (and masked) input vector
//	Ov  = V2   overflow scratch carried into the next word
func (s *ppcState) emitPackBodyPPC(fn *ppc64.Builder, bits int) {
	const acc, v, ov = "V0", "V1", "V2"
	word := 0
	off := 0
	haveAcc := false

	for k := 0; k < 32; k++ {
		// V = masked input vector k (src[4k..4k+3] at byte offset 16k).
		ppcLoadVec(fn, v, k)
		fn.Raw("VAND V15, %s, %s", v, v)

		if off == 0 {
			fn.Raw("VOR %s, %s, %s", v, v, acc) // acc = v (VOR x,x -> x)
			haveAcc = true
		} else {
			// acc |= v << off  (uses Ov=V2 as scratch)
			_ = ov
			s.shiftLeftOrPPC(fn, v, off, acc)
		}
		end := off + bits
		if end < 32 {
			off = end
			continue
		}
		// word full -> store acc.
		ppcStoreWord(fn, acc, word)
		word++
		if end > 32 {
			// acc = v >> (32-off)
			s.shiftRightIntoPPC(fn, v, 32-off, acc)
			haveAcc = true
			off = end - 32
		} else {
			haveAcc = false
			off = 0
		}
	}
	if haveAcc {
		ppcStoreWord(fn, acc, word)
	}
}

// shiftLeftOrPPC: acc |= v << n. Uses the hoisted count register for n (or a
// V14 reload if it spilled the pool), VSLW v<<cnt into Ov(V2), VOR into acc.
func (s *ppcState) shiftLeftOrPPC(fn *ppc64.Builder, v string, n int, acc string) {
	cnt := s.countVecReg(fn, n)
	fn.Raw("VSLW %s, %s, V2", v, cnt). // V2 = v << cnt (Plan9: data, count, dst)
		Raw("VOR V2, %s, %s", acc, acc)
}

// shiftRightIntoPPC: acc = v >> n.
func (s *ppcState) shiftRightIntoPPC(fn *ppc64.Builder, v string, n int, acc string) {
	cnt := s.countVecReg(fn, n)
	fn.Raw("VSRW %s, %s, %s", v, cnt, acc) // acc = v >> cnt (Plan9: data, count, dst)
}

// ppcLoadVec loads src vector k (byte offset 16k) into reg.
func ppcLoadVec(fn *ppc64.Builder, reg string, k int) {
	if k == 0 {
		fn.Raw("LXVW4X (R4), %s", vsOf(reg))
		return
	}
	fn.Raw("MOVD $%d, R8", 16*k).
		Raw("LXVW4X (R4)(R8), %s", vsOf(reg))
}

// ppcStoreWord stores reg to output word `word` (byte offset 16*word).
func ppcStoreWord(fn *ppc64.Builder, reg string, word int) {
	if word == 0 {
		fn.Raw("STXVW4X %s, (R3)", vsOf(reg))
		return
	}
	fn.Raw("MOVD $%d, R8", 16*word).
		Raw("STXVW4X %s, (R3)(R8)", vsOf(reg))
}

func ppcMain() {
	f := emit.NewFile("ppc64le")
	s := &ppcState{f: f, countSym: map[int]string{}}

	masks := map[int]string{}
	for bits := 1; bits <= 32; bits++ {
		masks[bits] = f.Data(fmt.Sprintf("vmask%d", bits), splat4(mask32(bits)))
	}
	for bits := 1; bits <= 32; bits++ {
		genPackPPC(s, bits, masks[bits])
		genUnpackPPC(s, bits, masks[bits])
	}

	if err := os.WriteFile("pack_ppc64le.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote pack_ppc64le.s")

	if err := os.WriteFile("pack_decl_ppc64le.go", []byte(genDeclsPPC()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote pack_decl_ppc64le.go")
}

func genDeclsPPC() string {
	var b strings.Builder
	b.WriteString("// Code generated by go-asmgen (pack_ppc64le_gen.go). DO NOT EDIT.\n\n")
	b.WriteString("//go:build ppc64le\n\npackage bitpack\n\n")
	for bits := 1; bits <= 32; bits++ {
		fmt.Fprintf(&b, "func packBits%d_VSX(dst []byte, src []uint32, blocks int)\n", bits)
		fmt.Fprintf(&b, "func unpackBits%d_VSX(dst []uint32, src []byte, blocks int)\n", bits)
	}
	b.WriteString("\nvar packVSXtab = [33]func(dst []byte, src []uint32, blocks int){\n")
	for bits := 1; bits <= 32; bits++ {
		fmt.Fprintf(&b, "\t%d: packBits%d_VSX,\n", bits, bits)
	}
	b.WriteString("}\n\nvar unpackVSXtab = [33]func(dst []uint32, src []byte, blocks int){\n")
	for bits := 1; bits <= 32; bits++ {
		fmt.Fprintf(&b, "\t%d: unpackBits%d_VSX,\n", bits, bits)
	}
	b.WriteString("}\n")
	return b.String()
}

func main() { ppcMain() }

func mask32(bits int) uint32 {
	if bits == 32 {
		return 0xffffffff
	}
	return uint32(1<<uint(bits)) - 1
}

// splat4 returns v little-endian-replicated into 4 uint32 words (16 bytes), the
// in-memory form an LXVW4X reads as [v,v,v,v].
func splat4(v uint32) []byte {
	b := make([]byte, 16)
	for i := 0; i < 4; i++ {
		b[i*4+0] = byte(v)
		b[i*4+1] = byte(v >> 8)
		b[i*4+2] = byte(v >> 16)
		b[i*4+3] = byte(v >> 24)
	}
	return b
}
