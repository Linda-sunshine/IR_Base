package topicmodels.correspondenceModels;

import structures.*;
import utils.Utils;

import java.util.Arrays;
import java.util.Collection;

/**
 * Created by jetcai1900 on 3/28/17.
 */
public class ACCTM_GlobalC extends ACCTM_C {
    double[] m_rareWordTopicSStat;
    double m_rareWordSStat;

    double[] m_rareWordTopicProb;

    public ACCTM_GlobalC(int number_of_iteration, double converge, double beta, _Corpus c, double lambda,
                         int number_of_topics, double alpha, double burnIn, int lag, double[] gamma){
        super(number_of_iteration, converge, beta, c, lambda,
         number_of_topics, alpha, burnIn,  lag, gamma);
    }

    public String toString(){
        return String.format("ACCTM_GlobalCtopic model [k:%d, alpha:%.2f, beta:%.2f, gamma1:%.2f, gamma2:%.2f, Gibbs Sampling]",
                number_of_topics, d_alpha, d_beta, m_gamma[0], m_gamma[1]);
    }

    protected void initialize_probability(Collection<_Doc> collection){
        createSpace();

        for(int i=0; i<number_of_topics; i++)
            Arrays.fill(word_topic_sstat[i], d_beta);

        Arrays.fill(m_sstat, d_beta*vocabulary_size);

        m_rareWordTopicSStat = new double[vocabulary_size];
        Arrays.fill(m_rareWordTopicSStat, d_beta);

        m_rareWordTopicProb = new double[vocabulary_size];
        Arrays.fill(m_rareWordTopicProb, 0);

        m_rareWordSStat = d_beta*vocabulary_size;

        for(_Doc d:collection){
            if(d instanceof _ParentDoc){
                d.setTopics4Gibbs(number_of_topics, 0);
                for(_Stn stnObj: d.getSentences())
                    stnObj.setTopicsVct(number_of_topics);

                for( _ChildDoc cDoc: ((_ParentDoc) d).m_childDocs){
                    _ChildDoc4BaseWithPhi cDocWithPhi = (_ChildDoc4BaseWithPhi)cDoc;
                    cDocWithPhi.createXSpace(number_of_topics, m_gamma.length, vocabulary_size, d_beta);
                    cDocWithPhi.setTopics4Gibbs(number_of_topics, 0);

                    for(_Word w: cDocWithPhi.getWords()){
                        int xid = w.getX();
                        int tid = w.getTopic();
                        int wid = w.getIndex();
                        //update global
                        if(xid==0){
                            word_topic_sstat[tid][wid] ++;
                            m_sstat[tid] ++;
                        }else{
                            m_rareWordTopicSStat[wid] ++;
                            m_rareWordSStat ++;
                        }
                    }
                }

                for (_Word w:d.getWords()) {
                    word_topic_sstat[w.getTopic()][w.getIndex()]++;
                    m_sstat[w.getTopic()]++;
                }
            }
        }

        imposePrior();
        m_statisticsNormalized = false;
    }

    protected void sampleInChildDoc(_Doc d) {
        _ChildDoc4BaseWithPhi cDoc = (_ChildDoc4BaseWithPhi)d;
        int wid, tid, xid;
        double normalizedProb;

        for(_Word w:cDoc.getWords()){
            wid = w.getIndex();
            tid = w.getTopic();
            xid = w.getX();

            if(xid==0){
                cDoc.m_xTopicSstat[xid][tid] --;
                cDoc.m_xSstat[xid] --;
                cDoc.m_wordXStat.put(wid, cDoc.m_wordXStat.get(wid)-1);
                if(m_collectCorpusStats){
                    word_topic_sstat[tid][wid] --;
                    m_sstat[tid] --;
                }
            }else if(xid==1){
                cDoc.m_xTopicSstat[xid][wid]--;
                cDoc.m_xSstat[xid] --;
                cDoc.m_childWordSstat --;
                m_rareWordTopicSStat[wid] --;
                m_rareWordSStat --;
            }

            normalizedProb = 0;
            double pLambdaZero = childXInDocProb(0, cDoc);
            double pLambdaOne = childXInDocProb(1, cDoc);

            for(tid=0; tid<number_of_topics; tid++){
                double pWordTopic = childWordByTopicProb(tid, wid);
                double pTopic = childTopicInDocProb(tid, cDoc);

                m_topicProbCache[tid] = pWordTopic*pTopic*pLambdaZero;
                normalizedProb += m_topicProbCache[tid];
            }

            double pWordTopic = childLocalWordByTopicProb(wid, cDoc);
            m_topicProbCache[tid] = pWordTopic*pLambdaOne;
            normalizedProb += m_topicProbCache[tid];

            normalizedProb *= m_rand.nextDouble();
            for(tid=0; tid<m_topicProbCache.length; tid++){
                normalizedProb -= m_topicProbCache[tid];
                if(normalizedProb<=0)
                    break;
            }

            if(tid==m_topicProbCache.length)
                tid --;

            if(tid<number_of_topics){
                xid = 0;
                w.setX(xid);
                w.setTopic(tid);
                cDoc.m_xTopicSstat[xid][tid]++;
                cDoc.m_xSstat[xid]++;

                if(cDoc.m_wordXStat.containsKey(wid)){
                    cDoc.m_wordXStat.put(wid, cDoc.m_wordXStat.get(wid)+1);
                }else{
                    cDoc.m_wordXStat.put(wid, 1);
                }

                if(m_collectCorpusStats){
                    word_topic_sstat[tid][wid] ++;
                    m_sstat[tid] ++;
                }

            }else if(tid==(number_of_topics)){
                xid = 1;
                w.setX(xid);
                w.setTopic(tid);
                cDoc.m_xTopicSstat[xid][wid]++;
                cDoc.m_xSstat[xid]++;
                cDoc.m_childWordSstat ++;
                m_rareWordTopicSStat[wid] ++;
                m_rareWordSStat ++;
            }
        }
    }

    protected double childLocalWordByTopicProb(int wid, _ChildDoc4BaseWithPhi d){
        return (m_rareWordTopicSStat[wid])
                / (m_rareWordSStat);
    }

    @Override
    public void calculate_M_step(int iter) {
        //literally we do not have M-step in Gibbs sampling
        if (iter>m_burnIn && iter%m_lag == 0) {
            //accumulate p(w|z)
            for(int v=0; v<this.vocabulary_size; v++) {
                for(int i=0; i<this.number_of_topics; i++) {
                    topic_term_probabilty[i][v] += word_topic_sstat[i][v]; // accumulate the samples during sampling iterations
                }
                m_rareWordTopicProb[v] += m_rareWordTopicSStat[v];
            }

            //accumulate p(z|d)
            for(_Doc d:m_trainSet)
                collectStats(d);
        }
    }

    protected void finalEst() {
        //estimate p(w|z) from all the collected samples
        for(int i=0; i<this.number_of_topics; i++)
            Utils.L1Normalization(topic_term_probabilty[i]);

        Utils.L1Normalization(m_rareWordTopicProb);

        //estimate p(z|d) from all the collected samples
        for(_Doc d:m_trainSet)
            estThetaInDoc(d);
    }
}
