import matplotlib.pyplot as plt
import numpy as np


density = []
flops = []
bw = []
with open("roofline/results.out") as rl_file:
	rl_file.readline()
	for line in rl_file:
		line = line.split()
		density.append(float(line[1]))
		flops.append(float(line[2]))
		bw.append(float(line[3]))

x = np.array(density)
y = np.array(flops)

z = np.polyfit(x[:50],y[:50],15)
p_sp = np.poly1d(z)

z = np.polyfit(x[50:],y[50:],15)
p_dp = np.poly1d(z)


''' plotting '''
ax1 = plt.subplot(111)
ax1.margins(0.05)

ax1.set_xscale('log',basex=2)
ax1.set_yscale('log',basey=2)

ax1.plot(density[:50],flops[:50],'b+',x[:50],p_sp(x[:50]),'r-')
ax1.plot([x[0],x[13]],[y[0],y[13]],'g-',[x[13],x[49]],[y[13],y[49]],'g-')

ax1.grid(True,which='major',axis='both')
ax1.grid(True,which='minor',axis='both')
ax1.set_title("Roofline - Single Precision")
ax1.set_xlabel("Arithmetic Intensity (Flops/Byte)")
ax1.set_ylabel("GFlops")
#plt.savefig("sp_roofline.png",bbox_inches='tight',dpi=200)
plt.show()

plt.figure(2)
ax2 = plt.subplot(111)
ax2.margins(0.05)

ax2.loglog(density[50:],flops[50:],'b+-',x[50:],p_dp(x[50:]),'r-')

ax2.grid(True,which='major',axis='both')
ax2.grid(True,which='minor',axis='both')
ax2.set_title("Roofline - Double Precision")
ax2.set_xlabel("Arithmetic Intensity (Flops/Byte)")
ax2.set_ylabel("GFlops")

plt.show()
#plt.savefig("dp_roofline.png",bbox_inches='tight',dpi=200)

