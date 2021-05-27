import itertools
from models.model_fit import model_cnn, model_trans, model_rf_data, model_rf
#create df/grid of hparams for cnn
#dropout
c1 = [0.5,0.8]
#kernel reg
c2 = [1e-6,1e-8]
#nconv filters
c3 = [8,16,32]
#nconv stacks
c4 = [2,3,6]

#create df/grid of hparams for transformer
#dropout            
t1 = [0.5,0.8]
#nheads
t2 = [4,8]
#nlayers
t3 = [2,4,8]
#model depth
t4 = [128,256]
#layer norm
t5 = [True,False]
#ff neurons
t6 = [256,512]

#create df/grid of hparams for rf
#nestimators 
r1 = [10,50,100,200]
#maxfeat
r2 = ['auto', 'sqrt']
#min samples
r3 = [2,4,6]

#create grid
pcnn = list(itertools.product(c1,c2,c3,c4))
ptrans = list(itertools.product(t1,t2,t3,t4,t5,t6))
prf = list(itertools.product(r1,r2,r3))

#rf data
data = model_rf_data()

#rf
f = open("results/result_hparam_rf.txt", "w")
for x in prf:
    result = model_rf(x,data)
    line = "rf model - params:{} - performance:{}  \n".format(x,result)
    f.write(line)
f.close()

#cnn
f = open("results/result_hparam_cnn.txt", "w")
for x in pcnn:
    result = model_cnn(x)
    line = "cnn model - params:{} - performance:{}  \n".format(x,result)
    f.write(line)
f.close()

#trans
f = open("results/result_hparam_trans.txt", "w")
for x in ptrans:
    result = model_trans(x)
    line = "trans model - params:{} - performance:{}  \n".format(x,result)
    f.write(line)
f.close()