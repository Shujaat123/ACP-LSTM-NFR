import numpy as np
from keras.utils import to_categorical
from numpy import linalg as la

def load_sequences(path):
    path = path
    new_list=[]
    seq_list=[]
    lis = []
    lx=[]

    with open(path, 'r') as fp:
        for line in fp:
            if line[0] == '>':
                new_list.append(line[1:])

            else:
                seq = line[:-1]
                seq_list.append(seq)
        for i, item in enumerate(new_list):
            lis.append([item, seq_list[i]])
        for i in lis:
            if len(i[1])>60:
                x=([i[0],i[1][0:60]])
                lx.append(x)
            else:
                lx.append(i)
    return lx

def Isoelectric_Point(seq):
    AA = 'ACDEFGHIKLMNPQRSTVWY'
    x,y,z,w=1,2,4,8#16,2,1,1
    ip_val={
        'A':(6.11)*x  +(2.35)*y  +(9.87)*z,
        'C':(5.15)*x  +(1.92)*y  +(10.70)*z    +(8.37)*w,
        'D':(2.98)*x  +(1.99)*y  +(9.90)*z     +(3.90)*w,
        'E':(3.08)*x  +(2.1)*y   +(9.47)*z     +(4.07)*w,
        'F':(5.76)*x  +(2.2)*y   +(9.31)*z,
        'G':(6.06)*x  +(2.5)*y   +(9.78)*z,
        'H':(7.64)*x  +(1.8)*y   +(9.33)*z     +(6.04)*w,
        'I':(6.04)*x  +(2.32)*y  +(9.74)*z,
        'K':(9.47)*x  +(2.16)*y  +(9.06)*z     +(10.54)*w,
        'L':(6.04)*x  +(2.33)*y  +(9.74)*z,
        'M':(5.71)*x  +(2.13)*y  +(9.28)*z,
        'N':(5.43)*x  +(2.14)*y  +(8.72)*z,
        'P':(6.30)*x  +(1.95)*y  +(10.64)*z,
        'Q':(5.65)*x  +(2.17)*y  +(9.13)*z,
        'R':(10.76)*x +(1.82)*y  +(8.99)*z     +(12.48)*w,
        'S':(5.70)*x  +(2.19)*y  +(9.21)*z,
        'T':(5.60)*x  +(2.09)*y  +(9.10)*z,
        'V':(6.02)*x  +(2.29)*y  +(9.74)*z,
        'W':(5.88)*x  +(2.46)*y  +(9.76)*z,
        'Y':(5.63)*x  +(2.2)*y   +(9.21)*z     +(10.46)*w,
    }

    ip={}
    for aa1 in AA:
        ip[aa1] = 0
    sum=0
    for aa in seq:
        sum+=1
        ip[aa]+=ip_val[aa]

    ip_final=[]
    for aa in ip:
        ip_final.append(ip[aa]/sum)


    return ip_final

def proposed_features(train_seq, gap1=8,gap2=4):
    cksscpfea = []
    seq_label = []
    ip_feature=[]
    for sseq in train_seq:
        temp= CKSSCP([sseq], gap1=gap1,gap2=gap2)
        cksscpfea.append(temp[1][1:])
        seq_label.append(sseq[0])
        ip_feature.append(Isoelectric_Point(sseq[1]))


    x = np.array(cksscpfea)
    y = np.array(seq_label)

    ip = np.array(ip_feature)
    x = np.concatenate((x, ip), axis=1)
    return x,y


def SC_feature(aPair,bPair):

    A1=['R','K','H']                    #amino acid with electrically charged side chain [positive]
    A2=['D','E']                        #amino acid with electrically charged side chain [negative]
    B=['S','T','N','Q']                 #amino acid with polar uncharged side chain
    C=['C','G','P']                     #amino acid with special cases
    D=['A','I','L','M','F','W','Y','V'] #amino acid with hydrophobic side chain


    if aPair in A1:
        A='1'
    elif aPair in A2:
        A='1'
    elif aPair in B:
        A='3'
    elif aPair in C:
        A='4'
    elif aPair in D:
        A='5'

    if bPair in A1:
        B='1'
    elif bPair in A2:
        B='1'
    elif bPair in B:
        B='3'
    elif bPair in C:
        B='4'
    elif bPair in D:
        B='5'

    return A,B

def Charge_feature(aPair,bPair):

    A1=['R','K','H']                    #amino acid with electrically charged side chain [positive]
    A2=['D','E']                        #amino acid with electrically charged side chain [negative]

    charge=0
    valid=0
    if aPair in A1:
        charge+=1
        valid+=1
    elif aPair in A2:
        charge-=1
        valid+=1

    if bPair in A1:
        charge+=1
        valid+=1
    elif bPair in A2:
        charge-=1
        valid+=1

    return charge, valid

def minSequenceLength(fastas):
    minLen = 10000
    for i in fastas:
        if minLen > len(i[1]):
            minLen = len(i[1])
    return minLen

def CKSSCP(fastas, gap1=5,gap2=4, **kw):
    if gap1 < 0:
        print('Error: the gap should be equal or greater than zero' + '\n\n')
        return 0

    if minSequenceLength(fastas) < gap1+2:
        print('Error: all the sequence length should be larger than the (gap value) + 2 = ' + str(gap1+2) + '\n' + 'Current sequence length ='  + str(minSequenceLength(fastas)) + '\n\n')
        return 0

    if gap2 < 0:
        print('Error: the gap should be equal or greater than zero' + '\n\n')
        return 0

    if minSequenceLength(fastas) < gap2+2:
        print('Error: all the sequence length should be larger than the (gap value) + 2 = ' + str(gap2+2) + '\n' + 'Current sequence length ='  + str(minSequenceLength(fastas)) + '\n\n')
        return 0

    AA = 'ACDEFGHIKLMNPQRSTVWY'
    BB = '1345'
    CC = '12345'
    encodings = []
    bbPair = []
    ccPair = []

    for bb1 in BB:
        for bb2 in BB:
            bbPair.append(bb1 + bb2)
    for cc2 in CC:
        ccPair.append('1' + cc2 + 'C')
        ccPair.append('2' + cc2 + 'C')


    ccPair.append('3' + '1' + 'C')
    ccPair.append('3' + '2' + 'C')
    ccPair.append('4' + '1' + 'C')
    ccPair.append('4' + '2' + 'C')
    ccPair.append('5' + '1' + 'C')
    ccPair.append('5' + '2' + 'C')

    header = ['#']
    for g in range(gap1+1):
        for bb in bbPair:
            header.append(bb + '.gap' + str(g))

    for g in range(gap2+1):
        for cc in ccPair:
            header.append(cc + '.gap' + str(g))

    encodings.append(header)
    for i in fastas:
        name, sequence = i[0], i[1]
        code = [name]
        for g in range(gap1+1):
            myDict = {}
            for pair in bbPair:
                myDict[pair] = 0

            sum = 0
            for index1 in range(len(sequence)):
                index2 = index1 + g + 1
                if index1 < len(sequence) and index2 < len(sequence) and sequence[index1] in AA and sequence[index2] in AA:
                    A,B=SC_feature(sequence[index1],sequence[index2])
                    myDict[A + B] = myDict[A + B]+1

                    sum = sum + 1
            for pair in bbPair:
                code.append(myDict[pair])# / sum)

        for g in range(gap2+1):
            myDict = {}
            ccPair_count = {}
            for pair in ccPair:
                myDict[pair] = 0
                ccPair_count[pair] = 0
            sum = 0
            for index1 in range(len(sequence)):
                index2 = index1 + g + 1
                if index1 < len(sequence) and index2 < len(sequence) and sequence[index1] in AA and sequence[index2] in AA:
                    A,B=SC_feature(sequence[index1],sequence[index2])
                    charge,valid=Charge_feature(sequence[index1],sequence[index2])
                    if(valid>0):
                        myDict[A + B + 'C']+=charge
                        ccPair_count[A + B + 'C']+=1
                    sum = sum + 1
            for pair in ccPair:
                if(ccPair_count[pair]>0):
                    code.append(myDict[pair] / (sum*ccPair_count[pair]))
                else:
                    code.append(myDict[pair] / sum)

        encodings.append(code)
    return encodings

def TransDict_from_list(groups):
    transDict = dict()
    tar_list = ['0', '1', '2', '3', '4', '5', '6']
    result = {}
    index = 0
    for group in groups:
        g_members = sorted(group)  # Alphabetically sorted list
        for c in g_members:
            # print('c' + str(c))
            # print('g_members[0]' + str(g_members[0]))
            result[c] = str(tar_list[index])  # K:V map, use group's first letter as represent.
        index = index + 1
    return result

def get_3_protein_trids():
    nucle_com = []
    chars = ['0', '1', '2', '3', '4', '5', '6']
    base = len(chars)
    end = len(chars) ** 3
    for i in range(0, end):
        n = i
        ch0 = chars[n % base]
        n = n / base
        ch1 = chars[int(n % base)]
        n = n / base
        ch2 = chars[int(n % base)]
        nucle_com.append(ch0 + ch1 + ch2)
    return nucle_com

def translate_sequence(seq, TranslationDict):
    '''
    Given (seq) - a string/sequence to translate,
    Translates into a reduced alphabet, using a translation dict provided
    by the TransDict_from_list() method.
    Returns the string/sequence in the new, reduced alphabet.
    Remember - in Python string are immutable..

    '''
    import string
    from_list = []
    to_list = []
    for k, v in TranslationDict.items():
        from_list.append(k)
        to_list.append(v)
    # TRANS_seq = seq.translate(str.maketrans(zip(from_list,to_list)))
    TRANS_seq = seq.translate(str.maketrans(str(from_list), str(to_list)))
    # TRANS_seq = maketrans( TranslationDict, seq)
    return TRANS_seq

def get_4_nucleotide_composition(tris, seq, pythoncount=True):
    seq_len = len(seq)
    tri_feature = [0] * len(tris)
    k = len(tris[0])
    note_feature = [[0 for cols in range(len(seq) - k + 1)] for rows in range(len(tris))]
    if pythoncount:
        for val in tris:
            num = seq.count(val)
            tri_feature.append(float(num) / seq_len)
    else:
        # tmp_fea = [0] * len(tris)
        for x in range(len(seq) + 1 - k):
            kmer = seq[x:x + k]
            if kmer in tris:
                ind = tris.index(kmer)
                # tmp_fea[ind] = tmp_fea[ind] + 1
                note_feature[ind][x] = note_feature[ind][x] + 1
        # tri_feature = [float(val)/seq_len for val in tmp_fea]    #tri_feature type:list len:256
        u, s, v = la.svd(note_feature)
        for i in range(len(s)):
            tri_feature = tri_feature + u[i] * s[i] / seq_len
        # print tri_feature
        # pdb.set_trace()

    return tri_feature

def ACP_DL_feature(path):
    label = []
    interaction_pair = {}
    RNA_seq_dict = {}
    protein_seq_dict = {}
    protein_index = 0
    with open(path, 'r') as fp:
        for line in fp:
            if line[0] == '>':
                values = line[1:].strip().split('|')
                proteinName = values[0]
           
            else:
                seq = line[:-1]
                protein_seq_dict[protein_index] = seq
                protein_index = protein_index + 1
    # name_list = read_name_from_lncRNA_fasta('ncRNA-protein/lncRNA_RNA.fa')
    groups = ['AGV', 'ILFP', 'YMTS', 'HNQW', 'RK', 'DE', 'C']
    group_dict = TransDict_from_list(groups)
    protein_tris = get_3_protein_trids()
    # tris3 = get_3_trids()
    bpf=[]
    kmer=[]
    # get protein feature
    # pdb.set_trace()
    for i in protein_seq_dict:  # and protein_fea_dict.has_key(protein) and RNA_fea_dict.has_key(RNA):
        protein_seq = translate_sequence(protein_seq_dict[i], group_dict)
        bpf_feature = BPF(protein_seq_dict[i])
        # print('bpf:',shape(bpf_feature))
        # pdb.set_trace()
        # RNA_tri_fea = get_4_nucleotide_composition(tris, RNA_seq, pythoncount=False)
        protein_tri_fea = get_4_nucleotide_composition(protein_tris, protein_seq, pythoncount =False)

        bpf.append(bpf_feature)
        kmer.append(protein_tri_fea)
        # protein_index = protein_index + 1
        # chem_fea.append(chem_tmp_fea)
    return np.array(bpf), np.array(kmer)

def BPF(seq_temp):
    seq = seq_temp
    chars = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    fea = []
    tem_vec =[]
    k = 7
    for i in range(k):
        if seq[i] =='A':
            tem_vec = [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif seq[i]=='C':
            tem_vec = [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif seq[i]=='D':
            tem_vec = [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif seq[i]=='E':
            tem_vec = [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif seq[i]=='F':
            tem_vec = [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif seq[i]=='G':
            tem_vec = [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif seq[i]=='H':
            tem_vec = [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif seq[i]=='I':
            tem_vec = [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0]
        elif seq[i]=='K':
            tem_vec = [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0]
        elif seq[i]=='L':
            tem_vec = [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0]
        elif seq[i]=='M':
            tem_vec = [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0]
        elif seq[i]=='N':
            tem_vec = [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0]
        elif seq[i]=='P':
            tem_vec = [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0]
        elif seq[i]=='Q':
            tem_vec = [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0]
        elif seq[i]=='R':
            tem_vec = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0]
        elif seq[i]=='S':
            tem_vec = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0]
        elif seq[i]=='T':
            tem_vec = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0]
        elif seq[i]=='V':
            tem_vec = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0]
        elif seq[i]=='W':
            tem_vec = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]
        elif seq[i]=='Y':
            tem_vec = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]
        fea = fea + tem_vec
    return fea