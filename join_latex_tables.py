
check_assert = True
join_mode = "ABAB"
#join_mode = "AABB"

AABB_separator = None #" "

ABAB_A_pre = None #"\, "
ABAB_A_post = None
ABAB_B_pre = None
ABAB_B_post = None


s1 = "\\scriptsize{DIVINE}~\cite{moorthy2011blind}   & .922 & .921 & .988 & .923 & .888 & .917 \\\\" \
     "\\scriptsize{BLIINDS-II}~\cite{saad2012blind}  & .935 & .968 & .980  & .938 & .896 & .930  \\\\" \
     "\\scriptsize{BRISQUE}~\\cite{mittal2012no}      & .923 & .973 & .985 & .951 & .903 & .942 \\\\" \
     "\\scriptsize{CORNIA}~\\cite{ye2012unsupervised} & .951 & .965 & .987 & .968 & .917 & .935 \\\\" \
     "\\scriptsize{CNN}~\\cite{kang2014convolutional} & .953 & .981 & .984 & .953 & .933 & .953 \\\\" \
     "\\scriptsize{SOM}~\\cite{zhang2015som}          & .952 & .961 & .991 & .974 & .954 & .962 \\\\" \
     "\\scriptsize{BIECON}~\\cite{?}                  & .965 & .987 & .970 & .945 & .931 & .962 \\\\" \
     "\\scriptsize{PQR}~\\cite{?}                     & -- & -- & -- & -- & -- & .971\\\\" \
     "\\scriptsize{DNN}~\\cite{bosse2016deep}         & -- & -- & -- & -- & -- & .972 \\\\" \
     "\\scriptsize{RankIQA+FT}~\\cite{bagdanov2017}   & .975 & .986 & .994 & .988 & .960 & .982 \\\\" \
     "\\scriptsize{Hall.-IQA}~\\cite{?}       & .977 & .984 & .993 & .990 &  .960 & .982 \\\\" \
     "\\scriptsize{NSSADNN}~\\cite{?}        & -- & -- & -- & -- &  -- & \\textbf{.984} \\\\" \
     "\\scriptsize{GADA (our)}                       & \\textbf{.977}  & .978 & \\textbf{.994} & .968 & .943 & .973  \\\\"

s2 = "\\scriptsize{DIVINE}~\\cite{moorthy2011blind}   & .913 & .91  & .984 & .921 & .863 & .916 \\\\" \
     "\\scriptsize{BLIINDS-II}~\\cite{saad2012blind}  & .929 & .942 & .969 & .923 & .889 & .931 \\\\" \
     "\\scriptsize{BRISQUE}~\\cite{mittal2012no}      & .914 & .965 & .979 & .951 & .887 & .940  \\\\" \
     "\\scriptsize{CORNIA}~\\cite{ye2012unsupervised} & .943 & .955 & .976 & .969 & .906 & .942 \\\\" \
     "\\scriptsize{CNN}~\\cite{kang2014convolutional} & .952 & .977 & .978 & .962 & .908 & .956 \\\\" \
     "\\scriptsize{SOM}~\\cite{zhang2015som}          & .947 & .952 & .984 & .976 & .937 & .964 \\\\" \
     "\\scriptsize{BIECON}~\\cite{?}                  & .952 & .974 & .980 & .956 & .923 & .961\\\\" \
     "\\scriptsize{PQR}~\\cite{?}                     & -- & -- &-- & -- & -- & .965\\\\" \
     "\\scriptsize{DNN}~\\cite{bosse2016deep}         & -- & -- & -- & -- & -- & .960 \\\\" \
     "\\scriptsize{RankIQA+FT}~\\cite{bagdanov2017}  & .970 & .978 & .991 & .988 &  .954 & .981 \\\\" \
     "\\scriptsize{Hall.-IQA}~\\cite{?}       & .983 & .961 & .984 & .983 &  .989 & .982 \\\\" \
     "\\scriptsize{NSSADNN}~\\cite{?}       & -- & -- & -- & -- &  -- & \\textbf{.986} \\\\" \
     "\\scriptsize{GADA (our)}               & .963 & .948 & \\textbf{.991} & .958 &  .917 & .964  \\\\"



s1 = s1.split('\\\\')
s2 = s2.split('\\\\')
assert len(s1) == len(s2)

out = []
for l1, l2 in zip(s1, s2):
    cols1 = l1.split('&')
    cols2 = l2.split('&')
    name1 = cols1[0].lstrip().rstrip()
    name2 = cols2[0].lstrip().rstrip()
    if check_assert:
        if not name1 in name2 and not name2 in name1:
            raise ValueError(f"Name {name1} and {name2} are different")
    out_line = [name1]
    if join_mode is "ABAB":
        for c1, c2 in zip(cols1[1:], cols2[1:]):
            c1 = c1.strip()
            c2 = c2.strip()
            c1 = ABAB_A_pre + c1 if ABAB_A_pre is not None else c1
            c1 = c1 + ABAB_A_post if ABAB_A_post is not None else c1
            c2 = ABAB_B_pre + c2 if ABAB_B_pre is not None else c2
            c2 = ABAB_B_post + c2 if ABAB_B_post is not None else c2
            out_line.append(c1)
            out_line.append(c2)

    elif join_mode is "AABB":
        for c1 in cols1[1:]:
            out_line.append(c1.strip())
        if AABB_separator is not None:
            out_line.append(AABB_separator)
        for c2 in cols2[1:]:
            out_line.append(c2.strip())
    out.append(' & '.join(out_line))
out_s = ' \\\\\n'.join(out)

print(out_s)
