'''
Created on Jul 6, 2011

@author: lpinello
'''
import numpy as np
from numpy import log2, array, mean, zeros, hstack, vstack,digitize
from numpy.random import randint, rand, permutation
import cPickle
import time
import sys
from bioutilities import int2nt, Sequence


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


def calculate_mim_old(S,R,ng=Ngram(),non_redundant=True):

    if non_redundant:
        N_S,seq_lengths=ng.build_ngrams_fq_matrix(S)
        N_R,seq_lengths=ng.build_ngrams_fq_matrix(R)
    else:
        N_S,seq_lengths=ng.build_ngrams_fq_matrix(S,non_redundant=False)
        N_R,seq_lengths=ng.build_ngrams_fq_matrix(R,non_redundant=False)

    N_S=N_S/np.tile(seq_lengths.T,(1,N_S.shape[1]))
    N_R=N_R/np.tile(seq_lengths.T,(1,N_S.shape[1]))
    
    N_S=N_S/N_R.std(0)
    N_R=N_R/N_R.std(0)  
    
    P_S=N_S.sum(0)/N_S.sum()
    P_R=N_R.sum(0)/N_R.sum()
    
    return symmetrized_kldiv(P_S,P_R)
