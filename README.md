# DDOS Prevention in IoT Networks by Analyzing Sourc1e-Side Inter-Bot Traffic Using Machine Learning Techniques
This repository is the source code for the article *DDOS Prevention in IoT Networks by Analyzing Sourc1e-Side Inter-Bot Traffic Using Machine Learning Techniques* by Saba Malekzadeh, [Saleh Yousefi](http://facultystaff.urmia.ac.ir/s.yousefi/fa), and [Mir Saman Tajbakhsh](http://facultystaff.urmia.ac.ir/mirsamantajbakhsh/fa).

The source code is the extension for another article named *Deep Defence* available at [GitHub](https://github.com/santhisenan/DeepDefense).

# Publication
The paper is under review, hence no citation can be presented for now.

```Malekzadeh, S., Yousefi, S., & Tajbakhsh, M. S. (Year unpublished). "DDOS Prevention in IoT Networks by Analyzing Source-Side Inter-Bot Traffic Using Machine Learning Techniques." [Unpublished manuscript].```

# Data Format
In the article, three distinct scenarios are presented: *data*, *dataFull*, and *dataProbable*. The **data** scenario represents the main dataset utilized in the study, without any modifications. This scenario aligns with the approach employed in DeepDefence. On the other hand, **dataFull** corresponds to the dataset aggregated using the **AttackProfile** methodology, as outlined in the article. Lastly, the **dataProbable** scenario introduces a significant innovation within the article, whereby the prevention of a DDOS attack is achieved by leveraging probable knowledge about the initiation of an attack. This novel approach enhances the effectiveness of DDOS prevention strategies.

1. CSV Format for data:
```pkSeqID,stime,flgs,proto,saddr,sport,daddr,dport,pkts,bytes,state,ltime,seq,dur,mean,stddev,sum,min,max,spkts,dpkts,sbytes,dbytes,rate,srate,drate,attack,category,subcategory```

2. CSV Format for dataFull:
```pkSeqID,stime,flgs,proto,saddr,sport,daddr,dport,pkts,bytes,state,ltime,seq,dur,mean,stddev,sum,min,max,spkts,dpkts,sbytes,dbytes,rate,srate,drate,attack,category,subcategory,AttackProfile```

3. CSV Format for dataProbable:
```pkSeqID,stime,flgs,proto,saddr,sport,daddr,dport,pkts,bytes,state,ltime,seq,dur,mean,stddev,sum,min,max,spkts,dpkts,sbytes,dbytes,rate,srate,drate,attack,category,subcategory,AttackProfile,ProbableAttack```

# Licence
The authors of the article *DDOS Prevention in IoT Networks by Analyzing Source-Side Inter-Bot Traffic Using Machine Learning Techniques* (Saba Malekzadeh, Saleh Yousefi, and Mir Saman Tajbakhsh), have developed an implementation of their research on Github. To encourage further collaboration and knowledge sharing, the authors have made the source code available to others under the condition that proper citation of their paper is provided. By citing their work, researchers and developers can access and utilize the implementation on Github, fostering advancements in the field of IoT network security.