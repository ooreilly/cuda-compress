#pragma once 


            inline __device__ void opt5ds79_compute2(float * __restrict__ p_in, const int stride)
            {
float p_0 = p_in[stride * 0];
float p_1 = p_in[stride * 1];
float p_2 = p_in[stride * 2];
float p_3 = p_in[stride * 3];
float p_4 = p_in[stride * 4];
float p_5 = p_in[stride * 5];
float p_6 = p_in[stride * 6];
float p_7 = p_in[stride * 7];
float p_8 = p_in[stride * 8];
float p_9 = p_in[stride * 9];
float p_10 = p_in[stride * 10];
float p_11 = p_in[stride * 11];
float p_12 = p_in[stride * 12];
float p_13 = p_in[stride * 13];
float p_14 = p_in[stride * 14];
float p_15 = p_in[stride * 15];
float p_16 = p_in[stride * 16];
float p_17 = p_in[stride * 17];
float p_18 = p_in[stride * 18];
float p_19 = p_in[stride * 19];
float p_20 = p_in[stride * 20];
float p_21 = p_in[stride * 21];
float p_22 = p_in[stride * 22];
float p_23 = p_in[stride * 23];
float p_24 = p_in[stride * 24];
float p_25 = p_in[stride * 25];
float p_26 = p_in[stride * 26];
float p_27 = p_in[stride * 27];
float p_28 = p_in[stride * 28];
float p_29 = p_in[stride * 29];
float p_30 = p_in[stride * 30];
float p_31 = p_in[stride * 31];

               {
	        float acc1 = al4 * (p_4 + p_4);
                acc1 += al1 * (p_1 + p_1);
	        acc1 += al0 * p_0;
	        float acc2 = al3 * (p_3 + p_3);
	        acc2 += al2 * (p_2 + p_2);
	        p_in[0 * stride] = acc1 + acc2;
               }

               // High
               {
                const int nl = 16;
		float acc1 = ah3 * (p_2 + p_4);
		acc1 += ah0 * p_1;
		float acc2 = ah2 * (p_1 + p_3);
		acc2 += ah1 * (p_0 + p_2);
		p_in[(nl+0)*stride] = acc1 + acc2;
               }
    

               {
	        float acc1 = al4 * (p_2 + p_6);
                acc1 += al1 * (p_1 + p_3);
	        acc1 += al0 * p_2;
	        float acc2 = al3 * (p_1 + p_5);
	        acc2 += al2 * (p_0 + p_4);
	        p_in[1 * stride] = acc1 + acc2;
               }

               // High
               {
                const int nl = 16;
		float acc1 = ah3 * (p_0 + p_6);
		acc1 += ah0 * p_3;
		float acc2 = ah2 * (p_1 + p_5);
		acc2 += ah1 * (p_2 + p_4);
		p_in[(nl+1)*stride] = acc1 + acc2;
               }
    

               {
	        float acc1 = al4 * (p_0 + p_8);
                acc1 += al1 * (p_3 + p_5);
	        acc1 += al0 * p_4;
	        float acc2 = al3 * (p_1 + p_7);
	        acc2 += al2 * (p_2 + p_6);
	        p_in[2 * stride] = acc1 + acc2;
               }

               // High
               {
                const int nl = 16;
		float acc1 = ah3 * (p_2 + p_8);
		acc1 += ah0 * p_5;
		float acc2 = ah2 * (p_3 + p_7);
		acc2 += ah1 * (p_4 + p_6);
		p_in[(nl+2)*stride] = acc1 + acc2;
               }
    

               {
	        float acc1 = al4 * (p_2 + p_10);
                acc1 += al1 * (p_5 + p_7);
	        acc1 += al0 * p_6;
	        float acc2 = al3 * (p_3 + p_9);
	        acc2 += al2 * (p_4 + p_8);
	        p_in[3 * stride] = acc1 + acc2;
               }

               // High
               {
                const int nl = 16;
		float acc1 = ah3 * (p_4 + p_10);
		acc1 += ah0 * p_7;
		float acc2 = ah2 * (p_5 + p_9);
		acc2 += ah1 * (p_6 + p_8);
		p_in[(nl+3)*stride] = acc1 + acc2;
               }
    

               {
	        float acc1 = al4 * (p_4 + p_12);
                acc1 += al1 * (p_7 + p_9);
	        acc1 += al0 * p_8;
	        float acc2 = al3 * (p_5 + p_11);
	        acc2 += al2 * (p_6 + p_10);
	        p_in[4 * stride] = acc1 + acc2;
               }

               // High
               {
                const int nl = 16;
		float acc1 = ah3 * (p_6 + p_12);
		acc1 += ah0 * p_9;
		float acc2 = ah2 * (p_7 + p_11);
		acc2 += ah1 * (p_8 + p_10);
		p_in[(nl+4)*stride] = acc1 + acc2;
               }
    

               {
	        float acc1 = al4 * (p_6 + p_14);
                acc1 += al1 * (p_9 + p_11);
	        acc1 += al0 * p_10;
	        float acc2 = al3 * (p_7 + p_13);
	        acc2 += al2 * (p_8 + p_12);
	        p_in[5 * stride] = acc1 + acc2;
               }

               // High
               {
                const int nl = 16;
		float acc1 = ah3 * (p_8 + p_14);
		acc1 += ah0 * p_11;
		float acc2 = ah2 * (p_9 + p_13);
		acc2 += ah1 * (p_10 + p_12);
		p_in[(nl+5)*stride] = acc1 + acc2;
               }
    

               {
	        float acc1 = al4 * (p_8 + p_16);
                acc1 += al1 * (p_11 + p_13);
	        acc1 += al0 * p_12;
	        float acc2 = al3 * (p_9 + p_15);
	        acc2 += al2 * (p_10 + p_14);
	        p_in[6 * stride] = acc1 + acc2;
               }

               // High
               {
                const int nl = 16;
		float acc1 = ah3 * (p_10 + p_16);
		acc1 += ah0 * p_13;
		float acc2 = ah2 * (p_11 + p_15);
		acc2 += ah1 * (p_12 + p_14);
		p_in[(nl+6)*stride] = acc1 + acc2;
               }
    

               {
	        float acc1 = al4 * (p_10 + p_18);
                acc1 += al1 * (p_13 + p_15);
	        acc1 += al0 * p_14;
	        float acc2 = al3 * (p_11 + p_17);
	        acc2 += al2 * (p_12 + p_16);
	        p_in[7 * stride] = acc1 + acc2;
               }

               // High
               {
                const int nl = 16;
		float acc1 = ah3 * (p_12 + p_18);
		acc1 += ah0 * p_15;
		float acc2 = ah2 * (p_13 + p_17);
		acc2 += ah1 * (p_14 + p_16);
		p_in[(nl+7)*stride] = acc1 + acc2;
               }
    

               {
	        float acc1 = al4 * (p_12 + p_20);
                acc1 += al1 * (p_15 + p_17);
	        acc1 += al0 * p_16;
	        float acc2 = al3 * (p_13 + p_19);
	        acc2 += al2 * (p_14 + p_18);
	        p_in[8 * stride] = acc1 + acc2;
               }

               // High
               {
                const int nl = 16;
		float acc1 = ah3 * (p_14 + p_20);
		acc1 += ah0 * p_17;
		float acc2 = ah2 * (p_15 + p_19);
		acc2 += ah1 * (p_16 + p_18);
		p_in[(nl+8)*stride] = acc1 + acc2;
               }
    

               {
	        float acc1 = al4 * (p_14 + p_22);
                acc1 += al1 * (p_17 + p_19);
	        acc1 += al0 * p_18;
	        float acc2 = al3 * (p_15 + p_21);
	        acc2 += al2 * (p_16 + p_20);
	        p_in[9 * stride] = acc1 + acc2;
               }

               // High
               {
                const int nl = 16;
		float acc1 = ah3 * (p_16 + p_22);
		acc1 += ah0 * p_19;
		float acc2 = ah2 * (p_17 + p_21);
		acc2 += ah1 * (p_18 + p_20);
		p_in[(nl+9)*stride] = acc1 + acc2;
               }
    

               {
	        float acc1 = al4 * (p_16 + p_24);
                acc1 += al1 * (p_19 + p_21);
	        acc1 += al0 * p_20;
	        float acc2 = al3 * (p_17 + p_23);
	        acc2 += al2 * (p_18 + p_22);
	        p_in[10 * stride] = acc1 + acc2;
               }

               // High
               {
                const int nl = 16;
		float acc1 = ah3 * (p_18 + p_24);
		acc1 += ah0 * p_21;
		float acc2 = ah2 * (p_19 + p_23);
		acc2 += ah1 * (p_20 + p_22);
		p_in[(nl+10)*stride] = acc1 + acc2;
               }
    

               {
	        float acc1 = al4 * (p_18 + p_26);
                acc1 += al1 * (p_21 + p_23);
	        acc1 += al0 * p_22;
	        float acc2 = al3 * (p_19 + p_25);
	        acc2 += al2 * (p_20 + p_24);
	        p_in[11 * stride] = acc1 + acc2;
               }

               // High
               {
                const int nl = 16;
		float acc1 = ah3 * (p_20 + p_26);
		acc1 += ah0 * p_23;
		float acc2 = ah2 * (p_21 + p_25);
		acc2 += ah1 * (p_22 + p_24);
		p_in[(nl+11)*stride] = acc1 + acc2;
               }
    

               {
	        float acc1 = al4 * (p_20 + p_28);
                acc1 += al1 * (p_23 + p_25);
	        acc1 += al0 * p_24;
	        float acc2 = al3 * (p_21 + p_27);
	        acc2 += al2 * (p_22 + p_26);
	        p_in[12 * stride] = acc1 + acc2;
               }

               // High
               {
                const int nl = 16;
		float acc1 = ah3 * (p_22 + p_28);
		acc1 += ah0 * p_25;
		float acc2 = ah2 * (p_23 + p_27);
		acc2 += ah1 * (p_24 + p_26);
		p_in[(nl+12)*stride] = acc1 + acc2;
               }
    

               {
	        float acc1 = al4 * (p_22 + p_30);
                acc1 += al1 * (p_25 + p_27);
	        acc1 += al0 * p_26;
	        float acc2 = al3 * (p_23 + p_29);
	        acc2 += al2 * (p_24 + p_28);
	        p_in[13 * stride] = acc1 + acc2;
               }

               // High
               {
                const int nl = 16;
		float acc1 = ah3 * (p_24 + p_30);
		acc1 += ah0 * p_27;
		float acc2 = ah2 * (p_25 + p_29);
		acc2 += ah1 * (p_26 + p_28);
		p_in[(nl+13)*stride] = acc1 + acc2;
               }
    

               {
	        float acc1 = al4 * (p_24 + p_30);
                acc1 += al1 * (p_27 + p_29);
	        acc1 += al0 * p_28;
	        float acc2 = al3 * (p_25 + p_31);
	        acc2 += al2 * (p_26 + p_30);
	        p_in[14 * stride] = acc1 + acc2;
               }

               // High
               {
                const int nl = 16;
		float acc1 = ah3 * (p_26 + p_30);
		acc1 += ah0 * p_29;
		float acc2 = ah2 * (p_27 + p_31);
		acc2 += ah1 * (p_28 + p_30);
		p_in[(nl+14)*stride] = acc1 + acc2;
               }
    

               {
	        float acc1 = al4 * (p_26 + p_28);
                acc1 += al1 * (p_29 + p_31);
	        acc1 += al0 * p_30;
	        float acc2 = al3 * (p_27 + p_29);
	        acc2 += al2 * (p_28 + p_30);
	        p_in[15 * stride] = acc1 + acc2;
               }

               // High
               {
                const int nl = 16;
		float acc1 = ah3 * (p_28 + p_28);
		acc1 += ah0 * p_31;
		float acc2 = ah2 * (p_29 + p_29);
		acc2 += ah1 * (p_30 + p_30);
		p_in[(nl+15)*stride] = acc1 + acc2;
               }
    
 p_0 = p_in[stride * 0];
 p_1 = p_in[stride * 1];
 p_2 = p_in[stride * 2];
 p_3 = p_in[stride * 3];
 p_4 = p_in[stride * 4];
 p_5 = p_in[stride * 5];
 p_6 = p_in[stride * 6];
 p_7 = p_in[stride * 7];
 p_8 = p_in[stride * 8];
 p_9 = p_in[stride * 9];
 p_10 = p_in[stride * 10];
 p_11 = p_in[stride * 11];
 p_12 = p_in[stride * 12];
 p_13 = p_in[stride * 13];
 p_14 = p_in[stride * 14];
 p_15 = p_in[stride * 15];
 p_16 = p_in[stride * 16];
 p_17 = p_in[stride * 17];
 p_18 = p_in[stride * 18];
 p_19 = p_in[stride * 19];
 p_20 = p_in[stride * 20];
 p_21 = p_in[stride * 21];
 p_22 = p_in[stride * 22];
 p_23 = p_in[stride * 23];
 p_24 = p_in[stride * 24];
 p_25 = p_in[stride * 25];
 p_26 = p_in[stride * 26];
 p_27 = p_in[stride * 27];
 p_28 = p_in[stride * 28];
 p_29 = p_in[stride * 29];
 p_30 = p_in[stride * 30];
 p_31 = p_in[stride * 31];

               {
	        float acc1 = al4 * (p_4 + p_4);
                acc1 += al1 * (p_1 + p_1);
	        acc1 += al0 * p_0;
	        float acc2 = al3 * (p_3 + p_3);
	        acc2 += al2 * (p_2 + p_2);
	        p_in[0 * stride] = acc1 + acc2;
               }

               // High
               {
                const int nl = 8;
		float acc1 = ah3 * (p_2 + p_4);
		acc1 += ah0 * p_1;
		float acc2 = ah2 * (p_1 + p_3);
		acc2 += ah1 * (p_0 + p_2);
		p_in[(nl+0)*stride] = acc1 + acc2;
               }
    

               {
	        float acc1 = al4 * (p_2 + p_6);
                acc1 += al1 * (p_1 + p_3);
	        acc1 += al0 * p_2;
	        float acc2 = al3 * (p_1 + p_5);
	        acc2 += al2 * (p_0 + p_4);
	        p_in[1 * stride] = acc1 + acc2;
               }

               // High
               {
                const int nl = 8;
		float acc1 = ah3 * (p_0 + p_6);
		acc1 += ah0 * p_3;
		float acc2 = ah2 * (p_1 + p_5);
		acc2 += ah1 * (p_2 + p_4);
		p_in[(nl+1)*stride] = acc1 + acc2;
               }
    

               {
	        float acc1 = al4 * (p_0 + p_8);
                acc1 += al1 * (p_3 + p_5);
	        acc1 += al0 * p_4;
	        float acc2 = al3 * (p_1 + p_7);
	        acc2 += al2 * (p_2 + p_6);
	        p_in[2 * stride] = acc1 + acc2;
               }

               // High
               {
                const int nl = 8;
		float acc1 = ah3 * (p_2 + p_8);
		acc1 += ah0 * p_5;
		float acc2 = ah2 * (p_3 + p_7);
		acc2 += ah1 * (p_4 + p_6);
		p_in[(nl+2)*stride] = acc1 + acc2;
               }
    

               {
	        float acc1 = al4 * (p_2 + p_10);
                acc1 += al1 * (p_5 + p_7);
	        acc1 += al0 * p_6;
	        float acc2 = al3 * (p_3 + p_9);
	        acc2 += al2 * (p_4 + p_8);
	        p_in[3 * stride] = acc1 + acc2;
               }

               // High
               {
                const int nl = 8;
		float acc1 = ah3 * (p_4 + p_10);
		acc1 += ah0 * p_7;
		float acc2 = ah2 * (p_5 + p_9);
		acc2 += ah1 * (p_6 + p_8);
		p_in[(nl+3)*stride] = acc1 + acc2;
               }
    

               {
	        float acc1 = al4 * (p_4 + p_12);
                acc1 += al1 * (p_7 + p_9);
	        acc1 += al0 * p_8;
	        float acc2 = al3 * (p_5 + p_11);
	        acc2 += al2 * (p_6 + p_10);
	        p_in[4 * stride] = acc1 + acc2;
               }

               // High
               {
                const int nl = 8;
		float acc1 = ah3 * (p_6 + p_12);
		acc1 += ah0 * p_9;
		float acc2 = ah2 * (p_7 + p_11);
		acc2 += ah1 * (p_8 + p_10);
		p_in[(nl+4)*stride] = acc1 + acc2;
               }
    

               {
	        float acc1 = al4 * (p_6 + p_14);
                acc1 += al1 * (p_9 + p_11);
	        acc1 += al0 * p_10;
	        float acc2 = al3 * (p_7 + p_13);
	        acc2 += al2 * (p_8 + p_12);
	        p_in[5 * stride] = acc1 + acc2;
               }

               // High
               {
                const int nl = 8;
		float acc1 = ah3 * (p_8 + p_14);
		acc1 += ah0 * p_11;
		float acc2 = ah2 * (p_9 + p_13);
		acc2 += ah1 * (p_10 + p_12);
		p_in[(nl+5)*stride] = acc1 + acc2;
               }
    

               {
	        float acc1 = al4 * (p_8 + p_14);
                acc1 += al1 * (p_11 + p_13);
	        acc1 += al0 * p_12;
	        float acc2 = al3 * (p_9 + p_15);
	        acc2 += al2 * (p_10 + p_14);
	        p_in[6 * stride] = acc1 + acc2;
               }

               // High
               {
                const int nl = 8;
		float acc1 = ah3 * (p_10 + p_14);
		acc1 += ah0 * p_13;
		float acc2 = ah2 * (p_11 + p_15);
		acc2 += ah1 * (p_12 + p_14);
		p_in[(nl+6)*stride] = acc1 + acc2;
               }
    

               {
	        float acc1 = al4 * (p_10 + p_12);
                acc1 += al1 * (p_13 + p_15);
	        acc1 += al0 * p_14;
	        float acc2 = al3 * (p_11 + p_13);
	        acc2 += al2 * (p_12 + p_14);
	        p_in[7 * stride] = acc1 + acc2;
               }

               // High
               {
                const int nl = 8;
		float acc1 = ah3 * (p_12 + p_12);
		acc1 += ah0 * p_15;
		float acc2 = ah2 * (p_13 + p_13);
		acc2 += ah1 * (p_14 + p_14);
		p_in[(nl+7)*stride] = acc1 + acc2;
               }
    
 p_0 = p_in[stride * 0];
 p_1 = p_in[stride * 1];
 p_2 = p_in[stride * 2];
 p_3 = p_in[stride * 3];
 p_4 = p_in[stride * 4];
 p_5 = p_in[stride * 5];
 p_6 = p_in[stride * 6];
 p_7 = p_in[stride * 7];
 p_8 = p_in[stride * 8];
 p_9 = p_in[stride * 9];
 p_10 = p_in[stride * 10];
 p_11 = p_in[stride * 11];
 p_12 = p_in[stride * 12];
 p_13 = p_in[stride * 13];
 p_14 = p_in[stride * 14];
 p_15 = p_in[stride * 15];
 p_16 = p_in[stride * 16];
 p_17 = p_in[stride * 17];
 p_18 = p_in[stride * 18];
 p_19 = p_in[stride * 19];
 p_20 = p_in[stride * 20];
 p_21 = p_in[stride * 21];
 p_22 = p_in[stride * 22];
 p_23 = p_in[stride * 23];
 p_24 = p_in[stride * 24];
 p_25 = p_in[stride * 25];
 p_26 = p_in[stride * 26];
 p_27 = p_in[stride * 27];
 p_28 = p_in[stride * 28];
 p_29 = p_in[stride * 29];
 p_30 = p_in[stride * 30];
 p_31 = p_in[stride * 31];

               {
	        float acc1 = al4 * (p_4 + p_4);
                acc1 += al1 * (p_1 + p_1);
	        acc1 += al0 * p_0;
	        float acc2 = al3 * (p_3 + p_3);
	        acc2 += al2 * (p_2 + p_2);
	        p_in[0 * stride] = acc1 + acc2;
               }

               // High
               {
                const int nl = 4;
		float acc1 = ah3 * (p_2 + p_4);
		acc1 += ah0 * p_1;
		float acc2 = ah2 * (p_1 + p_3);
		acc2 += ah1 * (p_0 + p_2);
		p_in[(nl+0)*stride] = acc1 + acc2;
               }
    

               {
	        float acc1 = al4 * (p_2 + p_6);
                acc1 += al1 * (p_1 + p_3);
	        acc1 += al0 * p_2;
	        float acc2 = al3 * (p_1 + p_5);
	        acc2 += al2 * (p_0 + p_4);
	        p_in[1 * stride] = acc1 + acc2;
               }

               // High
               {
                const int nl = 4;
		float acc1 = ah3 * (p_0 + p_6);
		acc1 += ah0 * p_3;
		float acc2 = ah2 * (p_1 + p_5);
		acc2 += ah1 * (p_2 + p_4);
		p_in[(nl+1)*stride] = acc1 + acc2;
               }
    

               {
	        float acc1 = al4 * (p_0 + p_6);
                acc1 += al1 * (p_3 + p_5);
	        acc1 += al0 * p_4;
	        float acc2 = al3 * (p_1 + p_7);
	        acc2 += al2 * (p_2 + p_6);
	        p_in[2 * stride] = acc1 + acc2;
               }

               // High
               {
                const int nl = 4;
		float acc1 = ah3 * (p_2 + p_6);
		acc1 += ah0 * p_5;
		float acc2 = ah2 * (p_3 + p_7);
		acc2 += ah1 * (p_4 + p_6);
		p_in[(nl+2)*stride] = acc1 + acc2;
               }
    

               {
	        float acc1 = al4 * (p_2 + p_4);
                acc1 += al1 * (p_5 + p_7);
	        acc1 += al0 * p_6;
	        float acc2 = al3 * (p_3 + p_5);
	        acc2 += al2 * (p_4 + p_6);
	        p_in[3 * stride] = acc1 + acc2;
               }

               // High
               {
                const int nl = 4;
		float acc1 = ah3 * (p_4 + p_4);
		acc1 += ah0 * p_7;
		float acc2 = ah2 * (p_5 + p_5);
		acc2 += ah1 * (p_6 + p_6);
		p_in[(nl+3)*stride] = acc1 + acc2;
               }
    
 p_0 = p_in[stride * 0];
 p_1 = p_in[stride * 1];
 p_2 = p_in[stride * 2];
 p_3 = p_in[stride * 3];
 p_4 = p_in[stride * 4];
 p_5 = p_in[stride * 5];
 p_6 = p_in[stride * 6];
 p_7 = p_in[stride * 7];
 p_8 = p_in[stride * 8];
 p_9 = p_in[stride * 9];
 p_10 = p_in[stride * 10];
 p_11 = p_in[stride * 11];
 p_12 = p_in[stride * 12];
 p_13 = p_in[stride * 13];
 p_14 = p_in[stride * 14];
 p_15 = p_in[stride * 15];
 p_16 = p_in[stride * 16];
 p_17 = p_in[stride * 17];
 p_18 = p_in[stride * 18];
 p_19 = p_in[stride * 19];
 p_20 = p_in[stride * 20];
 p_21 = p_in[stride * 21];
 p_22 = p_in[stride * 22];
 p_23 = p_in[stride * 23];
 p_24 = p_in[stride * 24];
 p_25 = p_in[stride * 25];
 p_26 = p_in[stride * 26];
 p_27 = p_in[stride * 27];
 p_28 = p_in[stride * 28];
 p_29 = p_in[stride * 29];
 p_30 = p_in[stride * 30];
 p_31 = p_in[stride * 31];

               {
	        float acc1 = al4 * (p_2 + p_2);
                acc1 += al1 * (p_1 + p_1);
	        acc1 += al0 * p_0;
	        float acc2 = al3 * (p_3 + p_3);
	        acc2 += al2 * (p_2 + p_2);
	        p_in[0 * stride] = acc1 + acc2;
               }

               // High
               {
                const int nl = 2;
		float acc1 = ah3 * (p_2 + p_2);
		acc1 += ah0 * p_1;
		float acc2 = ah2 * (p_1 + p_3);
		acc2 += ah1 * (p_0 + p_2);
		p_in[(nl+0)*stride] = acc1 + acc2;
               }
    

               {
	        float acc1 = al4 * (p_2 + p_0);
                acc1 += al1 * (p_1 + p_3);
	        acc1 += al0 * p_2;
	        float acc2 = al3 * (p_1 + p_1);
	        acc2 += al2 * (p_0 + p_2);
	        p_in[1 * stride] = acc1 + acc2;
               }

               // High
               {
                const int nl = 2;
		float acc1 = ah3 * (p_0 + p_0);
		acc1 += ah0 * p_3;
		float acc2 = ah2 * (p_1 + p_1);
		acc2 += ah1 * (p_2 + p_2);
		p_in[(nl+1)*stride] = acc1 + acc2;
               }
    
 p_0 = p_in[stride * 0];
 p_1 = p_in[stride * 1];
 p_2 = p_in[stride * 2];
 p_3 = p_in[stride * 3];
 p_4 = p_in[stride * 4];
 p_5 = p_in[stride * 5];
 p_6 = p_in[stride * 6];
 p_7 = p_in[stride * 7];
 p_8 = p_in[stride * 8];
 p_9 = p_in[stride * 9];
 p_10 = p_in[stride * 10];
 p_11 = p_in[stride * 11];
 p_12 = p_in[stride * 12];
 p_13 = p_in[stride * 13];
 p_14 = p_in[stride * 14];
 p_15 = p_in[stride * 15];
 p_16 = p_in[stride * 16];
 p_17 = p_in[stride * 17];
 p_18 = p_in[stride * 18];
 p_19 = p_in[stride * 19];
 p_20 = p_in[stride * 20];
 p_21 = p_in[stride * 21];
 p_22 = p_in[stride * 22];
 p_23 = p_in[stride * 23];
 p_24 = p_in[stride * 24];
 p_25 = p_in[stride * 25];
 p_26 = p_in[stride * 26];
 p_27 = p_in[stride * 27];
 p_28 = p_in[stride * 28];
 p_29 = p_in[stride * 29];
 p_30 = p_in[stride * 30];
 p_31 = p_in[stride * 31];

               {
	        float acc1 = al4 * (p_0 + p_0);
                acc1 += al1 * (p_1 + p_1);
	        acc1 += al0 * p_0;
	        float acc2 = al3 * (p_1 + p_1);
	        acc2 += al2 * (p_0 + p_0);
	        p_in[0 * stride] = acc1 + acc2;
               }

               // High
               {
                const int nl = 1;
		float acc1 = ah3 * (p_0 + p_0);
		acc1 += ah0 * p_1;
		float acc2 = ah2 * (p_1 + p_1);
		acc2 += ah1 * (p_0 + p_0);
		p_in[(nl+0)*stride] = acc1 + acc2;
               }
    
 p_0 = p_in[stride * 0];
 p_1 = p_in[stride * 1];
 p_2 = p_in[stride * 2];
 p_3 = p_in[stride * 3];
 p_4 = p_in[stride * 4];
 p_5 = p_in[stride * 5];
 p_6 = p_in[stride * 6];
 p_7 = p_in[stride * 7];
 p_8 = p_in[stride * 8];
 p_9 = p_in[stride * 9];
 p_10 = p_in[stride * 10];
 p_11 = p_in[stride * 11];
 p_12 = p_in[stride * 12];
 p_13 = p_in[stride * 13];
 p_14 = p_in[stride * 14];
 p_15 = p_in[stride * 15];
 p_16 = p_in[stride * 16];
 p_17 = p_in[stride * 17];
 p_18 = p_in[stride * 18];
 p_19 = p_in[stride * 19];
 p_20 = p_in[stride * 20];
 p_21 = p_in[stride * 21];
 p_22 = p_in[stride * 22];
 p_23 = p_in[stride * 23];
 p_24 = p_in[stride * 24];
 p_25 = p_in[stride * 25];
 p_26 = p_in[stride * 26];
 p_27 = p_in[stride * 27];
 p_28 = p_in[stride * 28];
 p_29 = p_in[stride * 29];
 p_30 = p_in[stride * 30];
 p_31 = p_in[stride * 31];
}
