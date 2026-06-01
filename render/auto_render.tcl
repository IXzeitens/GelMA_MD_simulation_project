package require pbctools

set systems {"Gelatin" "Gel1MA" "Gel2MA" "Gel3MA"}
# Assuming only single replica in the production folder right now
set reps {""}

# Graphic settings
display resize 800 600
color Display Background white
axes location Off

set outDir "C:/Users/User/Desktop/Work/0510/production/plot/snapshot"
file mkdir $outDir

foreach sys $systems {
    foreach rep $reps {
        set dirName "${sys}${rep}"
        set outPrefix "${sys}${rep}"
        if {$rep == ""} {
            set outPrefix "${sys}_rep1"
        }
        
        set basePath "C:/Users/User/Desktop/Work/0510/production/${dirName}/Output"
        
        if {![file exists "$basePath/debug_1.psf"]} {
            puts "Skipping $dirName, PSF not found."
            continue
        }
        
        puts "Processing $dirName..."
        
        # Delete any existing molecules
        mol delete all
        
        # Load topology and initial structure
        mol new "$basePath/debug_1.psf"
        mol addfile "$basePath/debug_1.pdb"
        
        # Load DCDs in chronological order: NVT first, then all NPT parts auto-
        # discovered and sorted numerically (so part2 < part10 < part11). Adding
        # part5 / part6 / ... after extending the trajectory just works — no
        # need to edit this script every time auto_extend.py adds a new chunk.
        set dcdList [list]
        if {[file exists "$basePath/system_nvt.dcd"]} {
            lappend dcdList "system_nvt.dcd"
        }
        foreach f [lsort -dictionary [glob -nocomplain -directory $basePath -tails "system_npt_part*.dcd"]] {
            lappend dcdList $f
        }
        foreach dcd $dcdList {
            puts "  Loading $dcd..."
            mol addfile "$basePath/$dcd" waitfor all
        }
        
        # ---------- Representations ----------
        mol delrep 0 top

        # 1) Protein backbone: CPK, colored by chain
        mol representation CPK 1.0 0.3 12 12
        mol color Chain
        mol selection "protein"
        mol material Opaque
        mol addrep top

        # 2) LMA side chains: Licorice (thicker, stands out from protein),
        #    colored by chain so each chain's LMA matches its backbone color
        mol representation Licorice 0.4 12 12
        mol color Chain
        mol selection "resname LMA"
        mol material Opaque
        mol addrep top

        # 3) LMA reactive carbons (C8, C6) highlighted as larger VDW spheres
        #    in yellow so crosslink sites are visible at a glance
        mol representation VDW 0.7 20
        mol color ColorID 4
        mol selection "resname LMA and (name C8 or name C6)"
        mol material Opaque
        mol addrep top
        
        # ---------------- Initial Snapshot ----------------
        animate goto 0
        
        # Read initial box size from config.json to fix pbc missing in frame 0
        set fp [open "$basePath/../config.json" r]
        set file_data [read $fp]
        close $fp
        if {[regexp {"calculated_box_L_Angstrom":\s*([0-9.]+)} $file_data match boxL]} {
            pbc set "{$boxL $boxL $boxL}"
        }
        
        pbc wrap -center origin -centersel "protein" -compound fragment -all
        pbc box -off
        pbc box -center origin -color black -width 2
        
        # Rigorous camera lock: reset translation/rotation, center to origin, scale inversely to box size
        molinfo top set center_matrix "{ {1 0 0 0} {0 1 0 0} {0 0 1 0} {0 0 0 1} }"
        molinfo top set global_matrix "{ {1 0 0 0} {0 1 0 0} {0 0 1 0} {0 0 0 1} }"
        scale to [expr 1.5 / $boxL]
        
        display update
        
        set initial_out "$outDir/${outPrefix}_initial.bmp"
        render TachyonInternal $initial_out
        puts "  Saved $initial_out"
        
        # ---------------- Final Snapshot ----------------
        set num_frames [molinfo top get numframes]
        if {$num_frames > 0} {
            animate goto [expr $num_frames - 1]
            pbc wrap -center origin -centersel "protein" -compound fragment -all
            pbc box -off
            pbc box -center origin -color black -width 2
            
            # Read current frame's box size
            set boxList [pbc get]
            set finalBoxL [lindex [lindex $boxList 0] 0]
            if {$finalBoxL == 0} { set finalBoxL 80.0 }
            
            # Rigorous camera lock
            molinfo top set center_matrix "{ {1 0 0 0} {0 1 0 0} {0 0 1 0} {0 0 0 1} }"
            molinfo top set global_matrix "{ {1 0 0 0} {0 1 0 0} {0 0 1 0} {0 0 0 1} }"
            scale to [expr 1.5 / $finalBoxL]
            
            display update
            
            set final_out "$outDir/${outPrefix}_final.bmp"
            render TachyonInternal $final_out
            puts "  Saved $final_out"
        }
    }
}

puts "All rendering completed successfully!"
quit
