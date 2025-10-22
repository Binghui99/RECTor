# RECTor : Robust and efficient Attack on Tor

## How to run RecTor ?

0. - Download DeepCOFFEA dataset from [here](https://drive.google.com/file/d/1ZYFXfESD15SAR4Q8hsoVYdTHpTD8Orys/view?usp=sharing).
   The folder has files which are traces whose names are *circuit-index*_*site-index*. This mean, if files have same circuit_index in their names, they’re collected using the same circuit. Each line in the file consists of "*time_stamp*\t*packet_size*”. So you can split each line based on ‘\t’. For example, line.spilt(‘\t’)[0] is Time_stamp and line.spilt(‘\t’)[1] is pkt size. [Here](https://drive.google.com/drive/folders/1PG0sF6AHHn_2LxyoIztwjpoxDmB7r39z?usp=sharing) is window-separated data they already preprocessed and used in the paper (so with this data, you can **skip the bullet points, 1 and 2**)
   
   
   - Download timegap and 10k testing data from [here](https://drive.google.com/drive/folders/1JUC-KBghWX42yg19gYDcrospyuE16d6X?usp=sharing).
   
   - Download DeepCorr data from [here](https://drive.google.com/drive/folders/1Z4PyMCX99xME3T_LLvURejSfisP9jy4n?usp=sharing). 
