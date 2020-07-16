# for applying svm making new file of featues along with the target then we will classify with svm method


import os
filenames = os.listdir('D:\\final sem project\\dataset of coordinates\\3files1hand\\')
script_dir = os.path.dirname('D:\\final sem project\\dataset of coordinates\\3files1hand\\')

outfile = os.listdir('D:\\final sem project\\dataset of coordinates\\finalfileswithlabels\\')
direc = os.path.dirname('D:\\final sem project\\dataset of coordinates\\finalfileswithlabels\\')

count=0

for (fname,oname) in zip(filenames,outfile):
    abs_file_path = os.path.join(script_dir, fname)
    f = open(abs_file_path, "r").readlines()
    count=count+1
    
    if(count<=3):
        output = ["%s %s" % (item.strip(), 1) for item in f]
        
    elif(count>3 and count<=6):
        output = ["%s %s" % (item.strip(), 2) for item in f]
        
    elif(count>6 and count<=9):
        output = ["%s %s" % (item.strip(), 3) for item in f]
        
    elif(count>9 and count<=12):
        output = ["%s %s" % (item.strip(), 4) for item in f]
        
    elif(count>12 and count<=15):
        output = ["%s %s" % (item.strip(), 5) for item in f]
        
    elif(count>15 and count<=18):
        output = ["%s %s" % (item.strip(), 6) for item in f]
        
    else :
        output = ["%s %s" % (item.strip(), 7) for item in f]
    
    out_file_path = os.path.join(direc, oname)
    w = open(out_file_path, "w")
    w.write("\n".join(output))
    w.close()
