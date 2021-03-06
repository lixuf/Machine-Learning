import json
f_name="phaseB_1b_01.jsonF.json"
f=open(f_name,encoding='utf-8')
phase=json.load(f)
json.dump(
                phase,
                open('../PythonApplication15/phaseB_1b_01F.json', 'w'),
                indent=4
                
            )