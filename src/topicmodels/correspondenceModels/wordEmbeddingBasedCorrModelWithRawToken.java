package topicmodels.correspondenceModels;

import structures.*;
import utils.Utils;

import java.util.Arrays;
import java.util.Collection;

/**
 * Created by jetcai1900 on 4/4/17.
 *
 * reserve raw tokens
 */
public class wordEmbeddingBasedCorrModelWithRawToken extends wordEmbeddingBasedCorrModel{
    public wordEmbeddingBasedCorrModelWithRawToken(int number_of_iteration, double converge, double beta, _Corpus c,
                                                   double lambda, int number_of_topics, double alpha, double alpha_c,
                                                   double[]gamma, double burnIn, int lag){
        super(number_of_iteration, converge, beta, c, lambda, number_of_topics, alpha, alpha_c,
        gamma, burnIn, lag);
    }

    protected void initialize_probability(Collection<_Doc> collection){
        createSpace();

        m_xProbCache = new double[m_gamma.length];
        m_alpha_c = new double[number_of_topics];
        Arrays.fill(m_alpha_c, d_alpha_c);
        m_totalAlpha_c = Utils.sumOfArray(m_alpha_c);

        for(int i=0; i<number_of_topics; i++){
            Arrays.fill(word_topic_sstat[i], d_beta);
        }

        Arrays.fill(m_sstat, d_beta*vocabulary_size);

        for(_Doc d:collection){
            if(d instanceof _ParentDoc4WordEmbedding){
                _ParentDoc4WordEmbedding pDoc = (_ParentDoc4WordEmbedding)d;
                for(_Stn stnObj:pDoc.getSentences()){
                    stnObj.setTopicsVct(number_of_topics);
                }
//                d.setTopics4Gibbs(number_of_topics, 0);
                pDoc.setTopics4Gibbs(number_of_topics, 0, vocabulary_size, m_gamma.length);

                for(_ChildDoc cDoc:pDoc.m_childDocs) {
//                    cDoc.setTopics4Gibbs_LDA(number_of_topics, 0);
                    cDoc.createXSpace(number_of_topics, m_gamma.length);
                    cDoc.setTopics4Gibbs(number_of_topics, 0);
                    for(_Word w:cDoc.getWords()){
                        int wid = w.getIndex();
                        int tid = w.getTopic();
                        int xid = w.getX();

                        cDoc.m_sstat[tid] ++;
                        pDoc.m_commentThread_wordSS[xid][tid][wid]++;
                    }
                }
            }

            for(_Word w:d.getWords()){
                word_topic_sstat[w.getTopic()][w.getIndex()]++;
                m_sstat[w.getTopic()] ++;
            }
        }

        imposePrior();
        m_statisticsNormalized = false;
    }

    protected void sampleInChildDoc(_Doc d){
        int wid, tid, xid;
        _ChildDoc cDoc = (_ChildDoc)d;
        _ParentDoc4WordEmbedding pDoc = (_ParentDoc4WordEmbedding)(cDoc.m_parentDoc);

        for(_Word w:cDoc.getWords()){
            double normalizedProb = 0.0;
            wid = w.getIndex();
            tid = w.getTopic();
//            xid = 0;
            xid = w.getX();

            cDoc.m_sstat[tid]--;
            pDoc.m_commentThread_wordSS[xid][tid][wid]--;

            if(xid==0) {
                if (m_collectCorpusStats) {
                    word_topic_sstat[tid][wid]--;
                    m_sstat[tid] --;
                }
            }

            for(tid=0; tid<number_of_topics; tid++){
                double wordTopicProb = 0.0;
                if(xid==0){
                    wordTopicProb = wordByTopicProbInComm(wid, tid);
                }else{
                    wordTopicProb = wordByTopicEmbedInComm(w, tid, pDoc);
                }

                double topicProb = childTopicInDocProb(tid, cDoc);

                m_topicProbCache[tid] = wordTopicProb*topicProb;
                normalizedProb += m_topicProbCache[tid];
            }

            normalizedProb *= m_rand.nextDouble();
            for(tid=0; tid<number_of_topics; tid++){
                normalizedProb -= m_topicProbCache[tid];
                if(normalizedProb<0)
                    break;
            }

            if(tid==number_of_topics)
                tid --;

            w.setTopic(tid);
            cDoc.m_sstat[tid]++;

            pDoc.m_commentThread_wordSS[xid][tid][wid]++;
            if(xid==0) {
                if (m_collectCorpusStats) {
                    word_topic_sstat[tid][wid]++;
                    m_sstat[tid] ++;
                }
            }
        }
    }

    protected double wordByTopicEmbedInComm(_Word cWord, int tid, _ParentDoc pDoc){
        double wordEmbeddingSim = 0.0;

        double normalizedTerm = 0.0;

        int rawCWId = cWord.getRawIndex();

        for(_Word pWord:pDoc.getWords()){
            int pWId = pWord.getIndex();
            int pTId = pWord.getTopic();
            int rawPWId = pWord.getRawIndex();

            if(pTId!=tid)
                continue;

            double wordCosSim = m_wordSimMatrix[rawCWId][rawPWId];

            wordEmbeddingSim += wordCosSim;
            normalizedTerm += m_wordSimVec[rawPWId];
        }

        if(wordEmbeddingSim==0.0){
//            System.out.println("zero similarity for topic child\t"+tid);
        }

        wordEmbeddingSim += d_beta;
        normalizedTerm += d_beta*vocabulary_size;

        wordEmbeddingSim /= normalizedTerm;
        return wordEmbeddingSim;
    }
}
