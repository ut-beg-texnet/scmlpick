from picks2xml import main_picks as picks2xml

input = '/home/cam/EQCCT/sceqcct/eqcct-dev/results/eqcct/20240819T202427Z_20240819T202527Z'
output_file = 'picks.xml'
eqcctTHR = 0.001

picks2xml(input_file=input, output_file=output_file, ai='eqcc', thr_dict=eqcctTHR)
