'''
Created on Jul 6, 2011

@author: lpinello
'''
import numpy as np
from numpy import log2, array, mean, zeros, hstack, vstack,digitize
from numpy.random import randint, rand, permutation
from bioutilities import int2nt,Sequence
#from scipy.stats import rv_discrete, randint
import cPickle
import time
import sys
import mpi4py.rc
#mpi4py.rc.initialize=False
#mpi4py.rc.finalize=False
from  mpi4py import MPI


#helper  functions
def chunks(l, n):
    c=len(l)/n
    s=[l[i*c:i*c+c] for i in range(n-1)]
    return s+[l[(n-1)*c:]]

def randsample(p,k):
    e = hstack([0,p.cumsum()])
    e[-1]=1
    return digitize(rand(k),e)-1


class Ngram:
    def __init__(self,ngram_length=4,alphabet_size=4):
        #print 'Ngram initialization'
        #build useful dictionary
        self.alphabet_size=alphabet_size
        self.ngram_length=ngram_length
        
        all_ngram=map(self.int2ngram,range(self.alphabet_size**self.ngram_length),[self.ngram_length]*(self.alphabet_size**self.ngram_length),[self.alphabet_size]*(self.alphabet_size**self.ngram_length))
        self.ngram2index=dict()
        self.index2ngram=dict()
        self.non_redundant_ngram2index=dict()
        self.non_redundant_index2ngram=dict()
        self.ngram_rev_complement=dict()
        self.number_of_ngrams=len(all_ngram)    
        
        for idx,ngram in enumerate(all_ngram):
            self.ngram2index[ngram]=idx
            self.ngram_rev_complement[ngram]=Sequence.reverse_complement(ngram)
            self.index2ngram[idx]=ngram
        
        
        for idx,ngram in enumerate([all_ngram[i] for i in self.non_redundant_idx()]):
            self.non_redundant_ngram2index[ngram]=idx
            self.non_redundant_index2ngram[idx]=ngram
        
        self.number_of_non_redundant_ngrams=len(self.non_redundant_ngram2index)
                
    def int2ngram(self,idx,ngram_length,alphabet_size):
        l=[]
        for _ in range(ngram_length):
            l.append(int2nt[idx % alphabet_size])
            idx/=alphabet_size

        return "".join(l)[-1::-1]
    
    def non_redundant_idx(self):
        ngram_taken=set()
        non_redundant_idxs=[]
        for idx in range(self.number_of_ngrams):
            n=self.index2ngram[idx]
            if (self.ngram2index[n] in ngram_taken) or (self.ngram2index[self.ngram_rev_complement[n]] in ngram_taken ):
                pass
            else:
                non_redundant_idxs.append(idx)
                ngram_taken.add(self.ngram2index[n])
                ngram_taken.add(self.ngram2index[self.ngram_rev_complement[n]])
        
        return tuple(non_redundant_idxs)
    
    def build_ngram_fq_vector(self,seq):
   
        ngram_vector=zeros(self.number_of_ngrams)
   
        for i in xrange(len(seq)-self.ngram_length+1):
            try:
                ngram_vector[self.ngram2index[seq[i:i+self.ngram_length]]]+=1
            except:
                pass
            #ngram_vector[self.ngram2index[self.ngram_rev_complement[seq[i:i+self.ngram_length]]]]+=1
        
        return ngram_vector
    
    def build_ngram_fq_vector_non_redundant(self,seq):
   
        ngram_vector=zeros(self.number_of_non_redundant_ngrams)
   
        for i in xrange(len(seq)-self.ngram_length+1):
            
            try:
                ngram_vector[self.non_redundant_ngram2index[seq[i:i+self.ngram_length]]]+=1
            except:
                pass
            try:
                ngram_vector[self.non_redundant_ngram2index[self.ngram_rev_complement[seq[i:i+self.ngram_length]]]]+=1
            except:
                pass
        
        return ngram_vector
    
    def build_ngrams_fq_matrix(self,seq_set,non_redundant=True):
        if non_redundant:
            ngram_matrix=zeros((len(seq_set),self.number_of_non_redundant_ngrams))
            for idx_seq,seq in enumerate(seq_set):
                ngram_matrix[idx_seq,:]=self.build_ngram_fq_vector_non_redundant(seq)
        else:
            ngram_matrix=zeros((len(seq_set),self.number_of_ngrams))
            for idx_seq,seq in enumerate(seq_set):
                ngram_matrix[idx_seq,:]=self.build_ngram_fq_vector(seq)
        
        return ngram_matrix,np.array([len(seq) for seq in seq_set],ndmin=2)
    
     
    def build_ngrams_fq_matrix_mpi(self,seq_set,non_redundant=True,ncpu=2):
        #t1=time.clock()
        if ncpu==1:
            ngram_matrix=self.build_ngrams_fq_matrix(seq_set,non_redundant)
        else:
            #MPI.Init()
            comm = MPI.COMM_SELF.Spawn(sys.executable,args=['build_ngrams_fq_matrix_mpi.py'],maxprocs=ncpu)
            #print time.clock()-t1,'sec.', 'spawn'
            #parameters={'ng':copy.copy(self),'non_redundant':non_redundant}
            parameters={'ng':self.ngram_length,'non_redundant':non_redundant}
            #print parameters
            comm.bcast(parameters,root=MPI.ROOT)
            #print 'dopo:',parameters
            #print time.clock()-t1,'sec.', 'broadcast'
            S=chunks(seq_set,ncpu)
            #print S
            
            if non_redundant:
                counts=[len(c)*self.number_of_non_redundant_ngrams for c in S]
                full_matrix=zeros((len(seq_set),self.number_of_non_redundant_ngrams))
            else:
                counts=[len(c)*self.number_of_ngrams for c in S]
                full_matrix=zeros((len(seq_set),self.number_of_ngrams))
            
            #print time.clock()-t1,'sec.', 'division'   
            comm.scatter(S, root=MPI.ROOT)
            #print time.clock()-t1,'sec.', 'scatter'
            ngram_matrix=None
            comm.Gatherv(ngram_matrix,(full_matrix,(counts,None)),root=MPI.ROOT)
            #print time.clock()-t1,'sec.', 'gather'
            ngram_matrix=full_matrix
            comm.Disconnect()
            #MPI.Finalize()
        #print time.clock()-t1,'sec.', ' with:', ncpu,' cpu'
        return ngram_matrix
    
    
    def build_ngrams_profile_mpi(self,seq_set,non_redundant=True,ncpu=2):
        #t1=time.clock()
        if ncpu==1:
            profile=self.build_ngrams_fq_matrix(seq_set,non_redundant)
            profile=profile.sum(0)/profile.sum()
        else:
            #MPI.Init()
            comm = MPI.COMM_SELF.Spawn(sys.executable,args=['build_ngrams_fq_profile_mpi.py'],maxprocs=ncpu)
            #print time.clock()-t1,'sec.', 'spawn'
            #parameters={'ng':copy.copy(self),'non_redundant':non_redundant}
            parameters={'ng':self.ngram_length,'non_redundant':non_redundant}
            #print parameters
            comm.bcast(parameters,root=MPI.ROOT)
                        
            if non_redundant:
                profile = zeros(self.number_of_non_redundant_ngrams)
            else:
                profile=zeros(self.number_of_ngrams)
            
            #print 'dopo:',parameters
            #print time.clock()-t1,'sec.', 'broadcast'
            S=chunks(seq_set,ncpu)
            #print S
            
            #print time.clock()-t1,'sec.', 'division'   
            comm.scatter(S, root=MPI.ROOT)
            #print time.clock()-t1,'sec.', 'scatter'
            
            comm.Reduce(None,profile,op=MPI.SUM, root=MPI.ROOT)
            #print 'Profile:', profile, len(profile)
            
            #print 'somma:',profile.sum()
            #profile/=profile.sum()
            #print 'Profile:', profile

            #print time.clock()-t1,'sec.', 'reduce'
            comm.Disconnect()
            #MPI.Finalize()
        #print time.clock()-t1,'sec.', ' with:', ncpu,' cpu'
        return profile
    
    
    
    
    def save_to_file(self,filename=None):
        if not filename:
            filename='ng'+str(self.ngram_length)
        with open(filename,'wb+') as outfile:
            print 'saving...'
            cPickle.dump(self, outfile,2)
            print 'done'
        
    @classmethod
    def load_from_file(cls,filename):
        with open(filename,'rb') as infile:
            return cPickle.load(infile)


def kldiv(p,q,eps= np.finfo(float).eps):
    p=p+eps
    q=q+eps
    p/=p.sum()
    q/=q.sum()
    return sum(p*log2(p/q)) 

def symmetrized_kldiv(p,q,eps= np.finfo(float).eps):
    p=p+eps
    q=q+eps
    p/=p.sum()
    q/=q.sum()
    return mean( sum(p*log2(p/q)),sum(q*log2(q/p))) 

def calculate_pwm_bits(pwm,p_acgt=array([0.25,0.25,0.25,0.25]),eps= np.finfo(float).eps):
    e=0
    for i in range(pwm.shape[0]):
        pwm_p=pwm[i,:]+eps
        pwm_p/=pwm_p.sum()
        e+=sum(pwm_p*log2(pwm_p/p_acgt))

    return e

pwm_row = lambda e:[i  for i in array([1-3*e,e,e,e])[permutation(4)]]
generate_pwm= lambda e,n: vstack([pwm_row(e) for _ in range(n)])

def generate_motif_instance(pwm):
    motif=['']*pwm.shape[0]
    for position in range(pwm.shape[0]):
        #bg_model= rv_discrete(name='bg', values=([0, 1, 2,3], pwm[position,:]))
        #motif[position]=bg_model.rvs(size=1)[0]
        motif[position]=randsample(pwm[position,:],1)[0]
    return ''.join([int2nt[c] for c in motif])

def generate_random_sequence(length,p_acgt=array([0.25,0.25,0.25,0.25])):
    #bg_model= rv_discrete(name='bg', values=([0, 1, 2,3], p_acgt))
    #return ''.join([int2nt[c] for c in bg_model.rvs(size=length)])
    return ''.join([int2nt[c] for c in  randsample(p_acgt,length)]) #digitize rocks!

def generate_random_sequence_with_motif(length,pwm,p_acgt=array([0.25,0.25,0.25,0.25])):
    #bg_model= rv_discrete(name='bg', values=([0, 1, 2,3],p_acgt))
    #seq=''.join([int2nt[c] for c in bg_model.rvs(size=length)])
    seq=''.join([int2nt[c] for c in  randsample(p_acgt,length)]) #digitize rocks!
    motif=generate_motif_instance(pwm)
    st_point=randint(0,length-len(motif)+1)
    return ''.join([seq[0:st_point],motif,seq[st_point+len(motif):]])

def generate_sequence_set_random(number_of_sequences,length,p_acgt=array([0.25,0.25,0.25,0.25])):
    #seq_set=[None]*number_of_sequences
    seq_set=[]
    for _ in xrange(number_of_sequences):
        #seq_set[i]=generate_random_sequence(length,p_acgt)
        seq_set.append(generate_random_sequence(length,p_acgt))
    
    return seq_set

def generate_sequence_set_with_motif(number_of_sequences,length,pwm,p_acgt=array([0.25,0.25,0.25,0.25])):
    #seq_set=[None]*number_of_sequences
    seq_set=[]
    for _ in xrange(number_of_sequences):
        #seq_set[i]=generate_random_sequence_with_motif(length,pwm,p_acgt)
        seq_set.append(generate_random_sequence_with_motif(length,pwm,p_acgt))
  
    return seq_set

def generate_sequence_set_random_mpi(number_of_sequences,length,p_acgt=array([0.25,0.25,0.25,0.25]),ncpu=1):
    #t1=time.clock()   
    if ncpu==1:
        seq_set=generate_sequence_set_random(int(number_of_sequences),length,p_acgt)
    else:   
        #MPI.Init()
        number_of_sequences_to_process=number_of_sequences/(ncpu)
        parameters=hstack([number_of_sequences_to_process,length,p_acgt])
        
        comm = MPI.COMM_SELF.Spawn(sys.executable,args=['generate_seq_set_random_mpi.py'],maxprocs=ncpu-1)
        comm.Bcast(parameters, root=MPI.ROOT)
        
        number_of_sequences_to_process+=(number_of_sequences- (number_of_sequences/(ncpu))*(ncpu))
        
        #print number_of_sequences,number_of_sequences_to_process
        
        seq_set=generate_sequence_set_random(int(number_of_sequences_to_process),length,p_acgt)
        for i in range(0,ncpu-1):
            #print 'wait data from',i
            seq_set+=comm.recv(source=i,tag=99)
            #print 'received data from',i
        
        comm.Disconnect()
        #MPI.Finalize()
    #print 'Sequences generated:',len(seq_set)
    #print 'Time elapsed:',time.clock()-t1,' sec.'
    return seq_set

def generate_sequence_set_with_motif_mpi(number_of_sequences,length,pwm,p_acgt=array([0.25,0.25,0.25,0.25]),ncpu=1):
    #t1=time.clock()   
    if ncpu==1:
        seq_set=generate_sequence_set_with_motif(int(number_of_sequences),length,pwm,p_acgt)
    else:   
        #MPI.Init()
        #print pwm
        number_of_sequences_to_process=number_of_sequences/(ncpu)
        parameters=hstack([number_of_sequences_to_process,length,p_acgt,list(pwm.shape)])
        #print parameters
        comm = MPI.COMM_SELF.Spawn(sys.executable,args=['generate_seq_set_with_motif_mpi.py'],maxprocs=ncpu-1)
        comm.Bcast(parameters, root=MPI.ROOT)
        comm.Bcast(pwm, root=MPI.ROOT)
        number_of_sequences_to_process+=(number_of_sequences- (number_of_sequences/(ncpu))*(ncpu))
        
        #print number_of_sequences,number_of_sequences_to_process,pwm
        
        seq_set=generate_sequence_set_with_motif(int(number_of_sequences_to_process),length,pwm,p_acgt)
        for i in range(0,ncpu-1):
            #print 'wait data from',i
            seq_set+=comm.recv(source=i,tag=99)
            #print 'received data from',i
        
        comm.Disconnect()
        #MPI.Finalize()
    #print 'Sequences generated:',len(seq_set)
    #print 'Time elapsed:',time.clock()-t1,' sec.'
    return seq_set

def extract_sequence_set_random_from_genome(coordinates,genome):
    pass

def extract_sequence_set_from_genome(coordinates,genome):
    pass

def calculate_mim_old(S,R,ng=Ngram(),non_redundant=True,ncpu=1):
    if ncpu==1:
        if non_redundant:
            N_S,seq_lengths=ng.build_ngrams_fq_matrix(S)
            N_R,seq_lengths=ng.build_ngrams_fq_matrix(R)
        else:
            N_S,seq_lengths=ng.build_ngrams_fq_matrix(S,non_redundant=False)
            N_R,seq_lengths=ng.build_ngrams_fq_matrix(R,non_redundant=False)
            
    else:
        if non_redundant:
            N_S,seq_lengths=ng.build_ngrams_fq_matrix_mpi(S,ncpu=ncpu)
            N_R,seq_lengths=ng.build_ngrams_fq_matrix_mpi(R,ncpu=ncpu)
        else:
            N_S,seq_lengths=ng.build_ngrams_fq_matrix_mpi(S,non_redundant=False,ncpu=ncpu)
            N_R,seq_lengths=ng.build_ngrams_fq_matrix_mpi(R,non_redundant=False,ncpu=ncpu)

    #N_S=N_S[:,ng.non_redundant()]
    #N_R=N_R[:,ng.non_redundant()]
    
    #STD deviation normalization?





    
    N_S=N_S/np.tile(seq_lengths.T,(1,N_S.shape[1]))
    N_R=N_R/np.tile(seq_lengths.T,(1,N_S.shape[1]))
    
    N_S=N_S/N_R.std(0)
    N_R=N_R/N_R.std(0)  
    
    
    P_S=N_S.sum(0)/N_S.sum()
    P_R=N_R.sum(0)/N_R.sum()
    
    return symmetrized_kldiv(P_S,P_R)


def calculate_mim(S,R,ng=Ngram(),non_redundant=True,ncpu=1,symmetric=True):
    P_S=ng.build_ngrams_profile_mpi(S, non_redundant, ncpu)
    P_R=ng.build_ngrams_profile_mpi(R, non_redundant, ncpu)
    if symmetric:
        return symmetrized_kldiv(P_S,P_R)
    else:
        return kldiv(P_S,P_R)
    
            
#TESTING        

def main():

    ###PARAMETERS###
    print 'main'
    alphabet_size=4
    ngram_length=8
    seq_length=15
    n_seq=500
    print 'load'
    t1=time.clock()
    ng=Ngram.load_from_file('ng'+str(ngram_length))
    print time.clock()-t1,'sec.'
    print 'loaded'
    
    p_acgt=array([0.15,0.35,0.35,0.15])
    ep=0
    pwm=array([[ep,ep,ep,1-3.0*ep,ep,1-3.0*ep],
           [ep,ep,ep,ep,1-3.0*ep,ep],
           [ep,ep,1-3.0*ep,ep,ep,ep],
           [1-3.0*ep,1-3.0*ep,ep,ep,ep,ep,]]).T.copy()
           
    
    
    #seq_set= generate_sequence_set_random_mpi(10000,1000,p_acgt=array([0.25,0.25,0.25,0.25]),ncpu=1)
    R= generate_sequence_set_random_mpi(5000,1000,p_acgt=array([0.25,0.25,0.25,0.25]),ncpu=4)
    #seq_set= generate_sequence_set_with_motif_mpi(10000,1000,pwm,p_acgt=array([0.25,0.25,0.25,0.25]),ncpu=1)
    #print 'generiamo il seq set'
    S= generate_sequence_set_with_motif_mpi(5000,1000,pwm,p_acgt=array([0.25,0.25,0.25,0.25]),ncpu=4)
    #seq_set=['aaaatttt','aaaatttt','aaaatttt','aaaatttt']
    print 'seq set generato'
    #a=ng.build_ngrams_fq_matrix_mpi(S, True, ncpu=1)
   # a=ng.build_ngrams_profile_mpi(seq_set, True, ncpu=6)
   # b=ng.build_ngrams_profile_mpi(seq_set, True, ncpu=1)
    
    t1=time.clock()
    print 'new,sp:',calculate_mim(S,R,ng,True,1)
    print time.clock()-t1,'sec.\n\n'
    
    t1=time.clock()
    print 'new,mp:',calculate_mim(S,R,ng,True,4)
    print time.clock()-t1,'sec.\n\n'
    t1=time.clock()    
    print 'old,sp:',calculate_mim_old(S,R,ng,True,1)
    print time.clock()-t1,'sec.\n\n'    
    t1=time.clock()    
    print 'old,mp:',calculate_mim_old(S,R,ng,True,4)
    print time.clock()-t1,'sec.\n\n'    
    

    '''

    print calculate_pwm_bits(m_ref )
    
    
    S= generate_sequence_set_random_mpi(n_seq,10)
    print len(S)
    #print generate_motif_instance(m_ref)
    
    #print generate_random_sequence(length,p_acgt)
    #print generate_random_sequence_with_motif(length,m_ref,p_acgt)
    
    
    
    
    ng=Ngram(ngram_length,alphabet_size)
    #print ng.all_ngram
    
    seq=generate_random_sequence(seq_length,p_acgt)
    print seq
    print Sequence.reverse_complement(seq)
    print nonzero(ng.build_ngram_fq_vector(seq))[0]
    print ng.all_ngram
    ngrams=[ ng.index2ngram[i] for i in nonzero(ng.build_ngram_fq_vector(seq))[0]]
    print 'ngrams:',ngrams
    
    for ngram in ngrams:
        assert(ngram in seq or ngram in Sequence.reverse_complement(seq))
    
    print 'Non redundant ngrams:',ng.non_redundant()
    print ng.all_ngram
    print ng.non_redundant()

    for k in range(3,11):
        ng=Ngram(k)
        ng.save_to_file()
        print ng.ngram_length
    

    for k in range(3,11):
        t1=time.clock()
        ng=Ngram.load_from_file('ng'+str(k))
        print ng.ngram_length
        print time.clock()-t1,'sec.'
    
    '''



if __name__ == "__main__":
    main()


