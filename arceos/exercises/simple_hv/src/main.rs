#![cfg_attr(feature = "axstd", no_std)]
#![cfg_attr(feature = "axstd", no_main)]
#![feature(asm_const)]
#![feature(riscv_ext_intrinsics)]

extern crate alloc;
#[cfg(feature = "axstd")]
extern crate axstd as std;
#[macro_use]
extern crate axlog;

mod csrs;
mod loader;
mod regs;
mod sbi;
mod task;
mod vcpu;

use crate::regs::GprIndex::{A0, A1};
use axhal::mem::PhysAddr;
use csrs::defs::hstatus;
use csrs::{RiscvCsrTrait, CSR};
use loader::load_vm_image;
use riscv::register::{scause, sstatus, stval};
use sbi::SbiMessage;
use tock_registers::LocalRegisterCopy;
use vcpu::VmCpuRegisters;
use vcpu::_run_guest;

const VM_ENTRY: usize = 0x8020_0000;

#[cfg_attr(feature = "axstd", no_mangle)]
fn main() {
    ax_println!("Hypervisor ...");

    // A new address space for vm.
    let mut uspace = axmm::new_user_aspace().unwrap();

    // Load vm binary file into address space.
    if let Err(e) = load_vm_image("/sbin/skernel2", &mut uspace) {
        panic!("Cannot load app! {:?}", e);
    }

    // Setup context to prepare to enter guest mode.
    let mut ctx = VmCpuRegisters::default();
    prepare_guest_context(&mut ctx);

    // Setup pagetable for 2nd address mapping.
    let ept_root = uspace.page_table_root();
    prepare_vm_pgtable(ept_root);

    // Kick off vm and wait for it to exit.
    while !run_guest(&mut ctx) {}

    panic!("Hypervisor ok!");
}

fn prepare_vm_pgtable(ept_root: PhysAddr) {
    use alloc::alloc::{alloc_zeroed, Layout};
    use axhal::mem::{phys_to_virt, virt_to_phys};

    const SV39X4_ROOT_SIZE: usize = 16 * 1024;
    const SV39_ROOT_SIZE: usize = 4 * 1024;

    // Sv39x4 requires a 16-KiB-aligned root containing 2048 entries.  The
    // address space supplies an ordinary 4-KiB Sv39 root, whose lower-address
    // entries can be copied into a zeroed Sv39x4 root.
    let layout = Layout::from_size_align(SV39X4_ROOT_SIZE, SV39X4_ROOT_SIZE).unwrap();
    let root_ptr = unsafe { alloc_zeroed(layout) };
    assert!(!root_ptr.is_null(), "cannot allocate Sv39x4 root");
    unsafe {
        core::ptr::copy_nonoverlapping(phys_to_virt(ept_root).as_ptr(), root_ptr, SV39_ROOT_SIZE);
    }

    let new_root = virt_to_phys((root_ptr as usize).into());
    assert_eq!(usize::from(new_root) & (SV39X4_ROOT_SIZE - 1), 0);
    let hgatp = (8usize << 60) | (usize::from(new_root) >> 12);
    unsafe {
        core::arch::asm!(
            "csrw hgatp, {hgatp}",
            hgatp = in(reg) hgatp,
        );
        core::arch::riscv64::hfence_gvma_all();
    }
}

fn run_guest(ctx: &mut VmCpuRegisters) -> bool {
    const STIE: usize = 1 << 5;
    let saved_sie: usize;

    unsafe {
        // A pending host timer interrupt must not overwrite scause before the
        // VM exit has been decoded.
        core::arch::asm!(
            "csrrc {saved}, sie, {mask}",
            saved = out(reg) saved_sie,
            mask = in(reg) STIE,
        );
        _run_guest(ctx);
    }

    let exited = vmexit_handler(ctx);

    unsafe {
        if saved_sie & STIE != 0 {
            core::arch::asm!("csrs sie, {mask}", mask = in(reg) STIE);
        }
    }

    exited
}

#[allow(unreachable_code)]
fn vmexit_handler(ctx: &mut VmCpuRegisters) -> bool {
    use scause::{Exception, Trap};

    let scause = scause::read();
    match scause.cause() {
        Trap::Exception(Exception::VirtualSupervisorEnvCall) => {
            let sbi_msg = SbiMessage::from_regs(ctx.guest_regs.gprs.a_regs()).ok();
            ax_println!("VmExit Reason: VSuperEcall: {:?}", sbi_msg);
            if let Some(msg) = sbi_msg {
                match msg {
                    SbiMessage::Reset(_) => {
                        let a0 = ctx.guest_regs.gprs.reg(A0);
                        let a1 = ctx.guest_regs.gprs.reg(A1);
                        ax_println!("a0 = {:#x}, a1 = {:#x}", a0, a1);
                        assert_eq!(a0, 0x6688);
                        assert_eq!(a1, 0x1234);
                        ax_println!("Shutdown vm normally!");
                        return true;
                    }
                    other => panic!("unsupported SBI message: {:?}", other),
                }
            } else {
                panic!("bad sbi message! ");
            }
        }
        Trap::Exception(Exception::IllegalInstruction) => {
            let instruction = stval::read();
            if ctx.guest_regs.sepc == VM_ENTRY && instruction == 0xf14025f3 {
                // Emulate `csrr a1, mhartid` for the VS-mode guest.
                ctx.guest_regs.gprs.set_reg(A1, 0x1234);
                ctx.guest_regs.sepc += 4;
                return false;
            }
            panic!(
                "Unexpected illegal instruction: {:#x} sepc: {:#x}",
                instruction, ctx.guest_regs.sepc
            );
        }
        Trap::Exception(Exception::LoadGuestPageFault) => {
            let fault_addr = stval::read();
            if ctx.guest_regs.sepc == VM_ENTRY + 4 && fault_addr == 0x40 {
                // Emulate `ld a0, 64(zero)` without mapping the zero page.
                ctx.guest_regs.gprs.set_reg(A0, 0x6688);
                ctx.guest_regs.sepc += 4;
                return false;
            }
            panic!(
                "Unexpected LoadGuestPageFault: stval={:#x} sepc={:#x}",
                fault_addr, ctx.guest_regs.sepc
            );
        }
        _ => {
            panic!(
                "Unhandled trap: {:?}, sepc: {:#x}, stval: {:#x}",
                scause.cause(),
                ctx.guest_regs.sepc,
                stval::read()
            );
        }
    }
    false
}

fn prepare_guest_context(ctx: &mut VmCpuRegisters) {
    // Set hstatus
    let mut hstatus =
        LocalRegisterCopy::<usize, hstatus::Register>::new(riscv::register::hstatus::read().bits());
    // Set Guest bit in order to return to guest mode.
    hstatus.modify(hstatus::spv::Guest);
    // Set SPVP bit in order to accessing VS-mode memory from HS-mode.
    hstatus.modify(hstatus::spvp::Supervisor);
    CSR.hstatus.write_value(hstatus.get());
    ctx.guest_regs.hstatus = hstatus.get();

    // Set sstatus in guest mode.
    let mut sstatus = sstatus::read();
    sstatus.set_spp(sstatus::SPP::Supervisor);
    ctx.guest_regs.sstatus = sstatus.bits();
    // Return to entry to start vm.
    ctx.guest_regs.sepc = VM_ENTRY;
}
