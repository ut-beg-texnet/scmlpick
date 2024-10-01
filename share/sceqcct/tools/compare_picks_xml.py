import os
import sys
import xml.etree.ElementTree as ET
from datetime import datetime

def extract_picks_from_xml(file_path, source_name):
    """Extract picks from an XML file and organize them by station."""
    picks_by_station = {}
    
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    # Namespace handling
    namespace = {'ns': 'http://geofon.gfz-potsdam.de/ns/seiscomp3-schema/0.12'}
    
    # Find all picks
    for pick in root.findall('ns:EventParameters/ns:pick', namespace):
        time = pick.find('ns:time/ns:value', namespace).text
        waveform_id = pick.find('ns:waveformID', namespace)
        station = waveform_id.get('stationCode')
        
        if station not in picks_by_station:
            picks_by_station[station] = {'source': source_name, 'picks': []}
        
        pick_data = {
            'time': datetime.fromisoformat(time.replace('Z', '+00:00')),
            'filterID': pick.find('ns:filterID', namespace).text,
            'methodID': pick.find('ns:methodID', namespace).text,
            'phaseHint': pick.find('ns:phaseHint', namespace).text,
            'evaluationMode': pick.find('ns:evaluationMode', namespace).text,
            'creationTime': pick.find('ns:creationInfo/ns:creationTime', namespace).text
        }
        picks_by_station[station]['picks'].append(pick_data)
    
    return picks_by_station

def compare_picks(picks1, source1, picks2, source2):
    """Compare the time difference between the closest picks from two XML objects."""
    
    # Ensure Real-Time comes first and Playback second
    if source1.lower() == 'playback' and source2.lower() == 'real-time':
        # Swap picks and sources to maintain order: Real-Time, Playback
        picks1, picks2 = picks2, picks1
        source1, source2 = source2, source1
    
    all_stations = set(picks1.keys()).union(set(picks2.keys()))
    
    for station in all_stations:
        station_picks1 = sorted(picks1.get(station, {}).get('picks', []), key=lambda p: p['time'])
        station_picks2 = sorted(picks2.get(station, {}).get('picks', []), key=lambda p: p['time'])
        
        print(f"\nStation: {station}")
        print(f"Number of picks in {source1}: {len(station_picks1)}")
        print(f"Number of picks in {source2}: {len(station_picks2)}")
        
        time_differences = []
        
        # Compare each pick in picks1 (Real-Time) to find the closest pick in picks2 (Playback)
        for pick1 in station_picks1:
            closest_diff = float('inf')
            closest_pick2_time = None
            
            for pick2 in station_picks2:
                diff = abs((pick1['time'] - pick2['time']).total_seconds())
                if diff < closest_diff:
                    closest_diff = diff
                    closest_pick2_time = pick2['time']
            
            if closest_pick2_time is not None:
                time_differences.append((pick1['time'], closest_pick2_time, closest_diff))
        
        print(f"Time differences (in seconds) between closest picks:")
        for time1, time2, diff in time_differences:
            print(f"Pick in {source1}: {time1} | Closest Pick in {source2}: {time2} | Difference: {diff:.2f} seconds")


def main():
    pick_obs = []
    sources = []

    # Prompt the user for the directory containing the .xml files
    directory = input("Please enter the directory containing the .xml files: ")

    # Check if the directory exists
    if not os.path.isdir(directory):
        print(f"The directory {directory} does not exist.")
        return
    
    computername = input("What is the name of the computer being used?: ")

    # Extract the name of the header directory for naming the results file
    header_directory_name = os.path.basename(os.path.normpath(directory))
    results_file_name = f"{header_directory_name}_{computername}results.txt"
    results_file_path = os.path.join(directory, results_file_name)
    
    # Redirect print statements to a file
    with open(results_file_path, 'w') as results_file:
        original_stdout = sys.stdout  # Save a reference to the original standard output
        sys.stdout = results_file  # Redirect standard output to the file

        try:
            # Iterate over all files in the directory and read .xml files
            for filename in os.listdir(directory):
                if filename.endswith(".xml"):
                    file_path = os.path.join(directory, filename)
                    
                    # Determine the source based on filename
                    if 'rt' in filename.lower():
                        source_name = 'Real-Time'
                    elif 'playback' in filename.lower():
                        source_name = 'Playback'
                    else:
                        source_name = 'Unknown'
                    
                    # Read the .xml file
                    try:
                        picks = extract_picks_from_xml(file_path, source_name)
                        pick_obs.append(picks)
                        sources.append(source_name)
                        print(f"Successfully read {filename} into an object.")
                    except Exception as e:
                        print(f"Failed to read {filename}: {e}")
                        return
            
            # Compare picks between the two XML objects
            if len(pick_obs) >= 2:
                compare_picks(pick_obs[0], sources[0], pick_obs[1], sources[1])
            else:
                print("Not enough XML files to compare.")
        finally:
            sys.stdout = original_stdout  # Reset standard output back to the original state

if __name__ == "__main__":
    main()
