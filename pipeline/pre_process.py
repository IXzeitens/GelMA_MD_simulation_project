# Date: 2025.08.08
# ========================================
# Date       : 2025.08.22
# Purpose    : Spliting packmol.pdb and rename the segment ID
# Description: 
#   1. Rename the segID in original PDB (index 72~75)
#   2. Splitting the renamed pdb based on Seg ID
# File path  : cd C:/Users/user/Desktop/Lab/25August/psfgen
# VMD path   : cd C:/Users/user/AppData/Roaming/Microsoft/Windows/Start Menu/Programs/University of Illinois/VMD
# ========================================
# pre_process.py

# ======== Ranges Setting ========
ranges = [
    (1,     303,   'GA'),
    (304,  606,   'GB'),
    (607,  909,   'GC'),
    (910, 14712, 'WT3')
]

input_file = 'Gelatin_wb.pdb'                                          # write-in original PDB
output_prefix = ''            

def get_segname(atom_id):
    for start, end, seg in ranges:                                      # define (start, end, seg)
        if start <= atom_id <= end:
            return seg
    return 'UNKN'                                                       # fallback for unmatched atoms

def process_and_split(input_file, output_prefix=""):
    output_files = {}
    atom_id = 0

    with open(input_file, 'r') as f:
        for line in f:
            if line.startswith(('ATOM', 'HETATM')):
                atom_id += 1
                segname = get_segname(atom_id)                          # Modify the segID in index 72~75
                line = line.rstrip('\n').ljust(76)
                new_line = line[:72] + segname.ljust(4) + line[76:]
                segname_key = segname.strip()
                if segname_key not in output_files:
                    output_files[segname_key] = []
                output_files[segname_key].append(new_line + '\n')

    for segname, lines in output_files.items():
        outfile = f"{output_prefix}{segname}.pdb"
        with open(outfile, 'w') as out:
            out.writelines(lines)
        print(f" Output：{outfile} ({len(lines)} Lines)")

if __name__ == "__main__":
    process_and_split(input_file, output_prefix)
