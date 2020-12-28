import numpy as np

def declare(n, decl="float"):
    s = []
    for i in range(n):
        s += ['%s p_%d = p_in[stride * %d];' % (decl, i, i)]
    return "\n".join(s)

def header():
    return "inline __device__ void opt5ds79_compute2(float * __restrict__ p_in, const int stride) {"

def mirror(x, n):
    y = []
    for xi in x:
        z = xi
        if xi < 0:
            z = -xi
        if z >= n:
            z = 2 * n - 2 - z
        z = abs(z)
        if z >= n:
            z = 2 * n - 2 - z
        y.append(z)
    return y

def compute(i, n):
    idx = np.array(range(-4, 5))
    p = mirror(idx + 2*i, n)
    q = mirror(idx + 2*i + 1, n)
    p_m4, p_m3, p_m2, p_m1, p_00, p_p1, p_p2, p_p3, p_p4 = p
    q_m4, q_m3, q_m2, q_m1, q_00, q_p1, q_p2, q_p3, q_p4 = q
    #print(p, q)
    s = """
               {
	        float acc1 = al4 * (p_%d + p_%d);
                acc1 += al1 * (p_%d + p_%d);
	        acc1 += al0 * p_%d;
	        float acc2 = al3 * (p_%d + p_%d);
	        acc2 += al2 * (p_%d + p_%d);
	        p_in[%d] = acc1 + acc2;
               }

               // High
               {
                const int nl = %d;
		float acc1 = ah3 * (p_%d + p_%d);
		acc1 += ah0 * p_%d;
		float acc2 = ah2 * (p_%d + p_%d);
		acc2 += ah1 * (p_%d + p_%d);
		p_in[(nl+%d)*stride] = acc1 + acc2;
               }
    """ % (p_m4, p_p4, p_m1, p_p1, p_00, p_m3, p_p3, p_m2, p_p2, i, n // 2, q_m3, q_p3, q_00, q_m2, q_p2, q_m1,
            q_p1, i)
    return s


print(header())
n = 32
print(declare(n))
while n >= 4:
    for i in range(int(n)//2):
        print(compute(i, n))
    n = n // 2
#    print('printf("n = %d \\n");' % n)
#    print("print_array(p_in, 32, 1, 1);")
    print(declare(32, decl=""))
print("}")
