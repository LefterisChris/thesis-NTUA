#!/usr/bin/python
import sys

print "testing to see if i can execute a script within another script\n"
print sys.argv[1]

cores = 57
threads = int(sys.argv[1])
l = []

for i in range(1,cores+1):
	id = (i-1)*4 +1
	if id + threads - 1 == 228:
		l.append(str(id)+"-"+str(id+threads-2)+",0")
	else:
		l.append(str(id)+"-"+str(id+threads-1))
print l

proc_ids = ','.join(l) 
print proc_ids

y = 1

def function():
	y += 1
	print y


for i in range(1,10):
	function()