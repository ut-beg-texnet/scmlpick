import glob
import os

# Get a list of all .xml files in the current directory
xml_files = glob.glob("/home/cam/EQCCT/sceqcct/eqcct-dev/results/picks/*.xml")
l = ''

# Example command: Print each file name
for xml_file in xml_files:
    l += ' '
    l += xml_file

print(l)

# Example command: run a command using each file name
os.system(f"scxmlmerge  {l} > picks.xml")