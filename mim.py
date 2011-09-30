'''
Created on Sep 30, 2011

@author: lpinello
'''


'''
Created on Sep 14, 2011

@author: lpinello
'''
import os
from utils import calculate_mim_old as mim,Ngram
from bioutilities import Genome,Genome_mm, Coordinate
from numpy import zeros, inf
from numpy.random import  randint,permutation
from scipy.stats.kde import gaussian_kde
from progressbar import ProgressBar,Percentage
import argparse

parser = argparse.ArgumentParser(description='Calculate a measure of sequence specificity called Motif Independent Metric (MIM)')
parser.add_argument('bed_file',metavar='bed_file', type=str, nargs=1, help='a bed file containing the coordinates on the genome of reference')
parser.add_argument('genome_directory',metavar='genome_directory', type=str, nargs=1, help='directory containing the fasta files of the genome of reference')
parser.add_argument('--null_model', metavar='null_model_type', type=str, nargs=1, help='Type of null mode, available choices: shuffle, random_seq')
parser.add_argument('--null_rep', metavar='n_repetitions_null_model', type=int, nargs=1, help='Number of samples to generate the null model')
parser.add_argument('--genome_loading', metavar='loading_mechanism', type=str, nargs=1, help='Loading mechanism of the genome, available choices: simple (slower), memory_mapped (faster!)')


args = parser.parse_args()

alphabet_size=4
ngram_length=4
bed_file=args.bed_file[0]
genome_directory=args.genome_directory[0]
memory_mapped_genome=True 
n_repetitions_null_model=1000
shuffle=True

alphabet_size=4
ngram_length=4

if args.null_model:
    if args.null_model[0]== 'shuffle':
        shuffle=True
        print 'Using Shuffle Null model'
    elif args.null_model[0]== 'random_seq':
        shuffle=False    
    else:
        print 'Bad choice for null_model'
        parser.print_help()
        exit()

if args.genome_loading:
    if args.genome_loading[0]== 'memory_mapped':
        memory_mapped_genome=True
    elif args.genome_loading[0]== 'simple':
        memory_mapped_genome=False  
    else:
        print 'Bad choice for genome_loading'
        parser.print_help()
        exit()

if args.null_rep:
    try:
        if args.null_rep[0]>0:   
            n_repetitions_null_model=args.null_rep[0]
    except:
        print 'Bad choice for genome_loading'
        parser.print_help()
        exit()


print args

print '----------------------------------------------'
print '          Motif Independent Metric v0.1       '
print 'Please send any bugs to: lucapinello@gmail.com'
print '----------------------------------------------\n\n'
ng=Ngram(alphabet_size=4,ngram_length=4)    

print '>Loading genome from:',genome_directory

if memory_mapped_genome:
    g=Genome_mm(genome_directory)
else:
    g=Genome(genome_directory)

print 'Genome Loaded.'
    
mim_values=dict()
print '\n>Loading coordinates from:',bed_file
coordinates=Coordinate.bed_to_coordinates(bed_file)
print '%d coordinates loaded' % len(coordinates)

S=[]
R=[]

print '\n>Extracting sequences:'
pb = ProgressBar(widgets=['Sequences processed: ', Percentage()], maxval=len(coordinates)).start()
for idx,c in enumerate(coordinates):
    seq=g.extract_sequence(c)
    
    if 'n' not in seq:
        S.append(seq)
        
        if shuffle:
            seq_random=''.join( [S[-1][i] for i in permutation(len(S[-1] ))]   )
            
        else:
            has_n=True
            while has_n:
                random_bpstart=randint(1,g.chr_len[c.chr_id]-len(c)+1)
                c_random=Coordinate(c.chr_id,random_bpstart,random_bpstart+len(c)-1)
                seq_random=g.extract_sequence(c_random)
                if 'n' not in seq_random:
                    has_n=False
            
        R.append(seq_random)
    
    else:
        print 'Skipped sequence with Ns:',c
        coordinates.remove(c)
    pb.update(idx)

pb.finish()

print '\n>Building the null model (%d samples):' % n_repetitions_null_model
random_vs_random_values=zeros(n_repetitions_null_model)
pb = ProgressBar(widgets=['Null model generation: ', Percentage()], maxval=n_repetitions_null_model).start()
for k in xrange(n_repetitions_null_model):
    R1=[]

    for idx_seq,seq in enumerate(R):
        if shuffle:            
            seq_random= ''.join( [seq[i] for i in permutation(len(seq))]) 
        else:
            has_n=True
            while has_n:
                random_bpstart=randint(1,g.chr_len[coordinates[idx_seq].chr_id]-len(seq)+1)
                c_random=Coordinate(coordinates[idx_seq].chr_id,random_bpstart,random_bpstart+len(seq)-1)
                seq_random=g.extract_sequence(c_random)
                if 'n' not in seq_random:
                    has_n=False
            
        R1.append(seq_random)
    
    random_vs_random_values[k]=mim(R,R1,ng,True)
    pb.update(k)
pb.finish()    

null_model=gaussian_kde(random_vs_random_values)  
mim_value=mim(S,R,ng,True)
p_value = null_model.integrate_box(mim_value,inf)

print '\n\n[MIM value: %f with a p-value of:%f]' % (mim_value, p_value)        
   
    