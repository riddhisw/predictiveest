
KEYS = ['tc_7', 'tc_8', 'tc_10', 'tc_12','tc_14','st_1','st_2','st_3','st_4','st_6','st_7','st_8'] 
tcDict={ key_:[] for key_ in KEYS}

########################################
# REFERENCE DICTIONARY INPUT VALUES
########################################

tc_7 = [7, 7, 7, 7, 7] 
tc_8 = [8, 8, 8, 8, 8] 
tc_10 = [10, 10, 10, 10, 10] 
tc_12 = [12, 12, 12, 12, 12]
tc_14 = [14, 14, 14, 14, 14]

va_1 = [1, 2, 4, 6, 7] 

st_1 = [15, 15, 23, 23, 23]
st_2 = [24, 24, 24, 24, 24]
st_3 = st_2
st_4 = [18, 18, 18, 18, 18]
st_6 = [22, 21, 21, 21, 22]
st_7 = [20, 20, 20, 20, 20]
st_8 = [19, 19, 19, 19, 19]

va_2 = [13, 12, 11, 10, 9] 
va_3 = [8, 9, 4, 10, 11] 
va_4 = [1, 3, 5, 7, 8]
va_5 = [2, 1, 5, 8, 3]
va_6 = [1, 2, 3, 4, 5] 


tc_7_lbl = 'Nyquist r [dimless]'
tc_8_lbl = tc_7_lbl
tc_10_lbl = tc_7_lbl
tc_12_lbl = tc_7_lbl
tc_14_lbl = tc_7_lbl

st_1_lbl = r'$f_0 / \Delta\omega^B $'
st_2_lbl = r'$f_0 J / \Delta\omega^B_{max} $'
st_3_lbl = r'$ \Delta s$ '
st_4_lbl = r'Msmt Noise Lvl [% of 3 St. Dev $f_n$ Pts]'
st_6_lbl = st_2_lbl
st_7_lbl = st_3_lbl
st_8_lbl = st_4_lbl

tc_7_dial = [20., 10., 5., 2., 1.25] 
tc_8_dial = tc_7_dial
tc_10_dial = tc_7_dial
tc_12_dial = tc_7_dial
tc_14_dial = tc_7_dial 
st_1_dial = [2.0, 1.0, 0.998, 0.994, 0.99] 
st_2_dial = [0.1988, 0.3976, 0.7952, 1.1928, 1.988] 
st_3_dial = [1.8181818182, 0.9523809524, 0.487804878, 0.3278688525, 0.2469135802] 
st_4_dial = [0.01, 0.05, 0.1, 0.2, 0.25] 
st_6_dial = [0.1777777778, 0.3555555556, 0.8000000001, 1.0666666668, 1.4222222224]
st_7_dial = [0.9090909091, 0.4761904762, 0.3225806452, 0.243902439, 0.1960784314]
st_8_dial = st_4_dial

n_testbefore_1 = [0, 50, 25, 17, 13, 10, 5, 3 ] 
n_testbefore_2 = [0, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50]

n_predict_1 = [0, 100, 50, 33, 25, 20, 10, 7] 
n_predict_2 = n_testbefore_2
n_predict_3 = [0, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]


########################################
# REFERENCE DICTIONARY CREATE
########################################
tcDict['tc_7'] = [tc_7, va_1, tc_7_dial, tc_7_lbl, n_predict_1, n_testbefore_1]
tcDict['tc_8'] = [tc_8, va_1, tc_8_dial, tc_8_lbl, n_predict_1, n_testbefore_1]
tcDict['tc_10'] = [tc_10, va_1, tc_10_dial, tc_10_lbl, n_predict_1, n_testbefore_1]
tcDict['tc_12'] = [tc_12, va_1, tc_12_dial, tc_12_lbl, n_predict_1, n_testbefore_1]
tcDict['tc_14'] = [tc_14, va_1, tc_14_dial, tc_14_lbl, n_predict_1, n_testbefore_1]
tcDict['st_1'] = [st_1, va_2 , st_1_dial , st_1_lbl, n_predict_2, n_testbefore_2]
tcDict['st_2'] = [st_2, va_1 , st_2_dial , st_2_lbl, n_predict_2, n_testbefore_2]
tcDict['st_3'] = [st_3, va_3 , st_3_dial , st_3_lbl, n_predict_2, n_testbefore_2]
tcDict['st_4'] = [st_4, va_4 , st_4_dial , st_4_lbl, n_predict_3, n_testbefore_2]
tcDict['st_6'] = [st_6, va_5 , st_6_dial , st_6_lbl, n_predict_3, n_testbefore_2]
tcDict['st_7'] = [st_7, va_6 , st_7_dial , st_7_lbl, n_predict_3, n_testbefore_2]
tcDict['st_8'] = [st_8, va_4 , st_8_dial , st_8_lbl, n_predict_3, n_testbefore_2]