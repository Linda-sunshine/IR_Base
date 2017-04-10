package topicmodels.correspondenceModels;

import structures.*;
import utils.Utils;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;

/**
 * Created by jetcai1900 on 4/4/17.
 *
 * reserve raw tokens
 */
public class wordEmbeddingBasedCorrModelWithRawToken extends wordEmbeddingBasedCorrModel{

    int m_rawFeatureSize;

    public wordEmbeddingBasedCorrModelWithRawToken(int number_of_iteration, double converge, double beta, _Corpus c,
                                                   double lambda, int number_of_topics, double alpha, double alpha_c,
                                                   double[]gamma, double burnIn, int lag){
        super(number_of_iteration, converge, beta, c, lambda, number_of_topics, alpha, alpha_c,
        gamma, burnIn, lag);
        m_rawFeatureSize = c.m_rawFeatureList.size();
        m_mu = 0.05;
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
                pDoc.setTopics4GibbsbyRawToken(number_of_topics, 0, vocabulary_size, m_rawFeatureSize, m_gamma.length);

                for(_ChildDoc cDoc:pDoc.m_childDocs) {
//                    cDoc.setTopics4Gibbs_LDA(number_of_topics, 0);
                    cDoc.createXSpace(number_of_topics, m_gamma.length);
                    cDoc.setTopics4GibbsbyRawToken(number_of_topics, 0);
                    cDoc.setMu(m_mu);
                    for(_Word w:cDoc.getWords()){
                        int wid = w.getIndex();
                        int tid = w.getTopic();
                        int xid = w.getX();
                        int rawWId = w.getRawIndex();

                        cDoc.m_sstat[tid] ++;
                        if(xid==1)
                            pDoc.m_commentThread_wordSS[xid][tid][rawWId]++;
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

    public String toString(){
        return String.format("word embedding based Corr model with raw tokens [k:%d, alpha:%.2f, beta:%.2f, Gibbs Sampling]",
                number_of_topics, d_alpha, d_beta);
    }

    public void loadWordSim4Corpus(String wordSimFileName){
        double simThreshold = 1;
        if(wordSimFileName == null||wordSimFileName.isEmpty()){
            return;
        }

        int rawFeatureSize = m_rawFeatureSize;

        try{
            if(m_wordSimMatrix==null) {
                m_wordSimMatrix = new double[rawFeatureSize][rawFeatureSize];
                m_wordSimVec = new double[rawFeatureSize];
            }

            double maxSim = -2;
            double minSim = 2;

            for(int v=0; v<rawFeatureSize; v++) {
                Arrays.fill(m_wordSimMatrix[v], 0);
                Arrays.fill(m_wordSimVec, 0);
            }

            String tmpTxt;
            String[] lineContainer;

            HashMap<String, Integer> featureNameIndex = new HashMap<String, Integer>();
            for(int i=0; i<rawFeatureSize; i++){
                featureNameIndex.put(m_corpus.m_rawFeatureList.get(i), i);
            }

            BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(wordSimFileName), "UTF-8"));

            ArrayList<String> featureList = new ArrayList<String>();

            boolean firstLineFlag = false;
            int lineIndex = 0;
            while((tmpTxt=br.readLine())!=null){
                tmpTxt = tmpTxt.trim();
                if(tmpTxt.isEmpty())
                    continue;

                lineContainer = tmpTxt.split("\t");
                if(firstLineFlag==false){
                    for(int i=0; i<lineContainer.length; i++){
                        featureList.add(lineContainer[i]);
                    }
                    firstLineFlag = true;
                }else{
                    int rowWId = featureNameIndex.get(featureList.get(lineIndex)); //translate string into int

                    for(int i=0; i<lineContainer.length; i++){
                        int colWId = featureNameIndex.get(featureList.get(i));
//                        m_wordSimVec[rowWId] += Double.parseDouble(lineContainer[i]);
                        m_wordSimMatrix[rowWId][colWId] = Double.parseDouble(lineContainer[i]);

                        if(m_wordSimMatrix[rowWId][colWId] > maxSim){
                            maxSim = m_wordSimMatrix[rowWId][colWId];
                        }else{
                            if(m_wordSimMatrix[rowWId][colWId] < minSim) {
                                minSim = m_wordSimMatrix[rowWId][colWId];
                            }
                        }
                    }

                    lineIndex ++;
                }
            }
            normalizeSim(maxSim, minSim, simThreshold);
        }catch (Exception e){
            e.printStackTrace();
        }
    }

    public void normalizeSim(double maxSim, double minSim, double threshold){
        for(int i=0; i<m_rawFeatureSize; i++) {
            m_wordSimVec[i] = 0;
            for (int j = 0; j < m_rawFeatureSize; j++) {
                double normalizedSim = (m_wordSimMatrix[i][j] - minSim) / (maxSim - minSim);

                if(normalizedSim< threshold)
                    m_wordSimMatrix[i][j] = 0;
                else {
                    m_wordSimMatrix[i][j] = normalizedSim;
                    m_wordSimVec[i] += normalizedSim;
                }
            }
        }
    }


    @Override
    protected void sampleInParentDoc(_Doc d){
        int wid, tid, rawWId;

        _ParentDoc4WordEmbedding pDoc = (_ParentDoc4WordEmbedding)d;

        for(_Word w:pDoc.getWords()){
            double normalizedProb = 0;

            wid = w.getIndex();
            tid = w.getTopic();
            rawWId = w.getRawIndex();

            pDoc.m_sstat[tid] --;
            word_topic_sstat[tid][wid]--;
            m_sstat[tid] --;


            for(int k=0; k<number_of_topics; k++){
                double wordTopicProbfromCommentWordEmbed = parentWordByTopicProbFromCommentWordEmbed(rawWId, tid, k, pDoc);
                double wordTopicProbInDoc = parentWordByTopicProb(k, wid)/parentWordByTopicProb(0, wid);
                double topicProbfromComment = parentChildInfluenceProb(k, pDoc);
                double topicProbInDoc = parentTopicInDocProb(k, pDoc);

                m_topicProbCache[k] = wordTopicProbfromCommentWordEmbed
                        *wordTopicProbInDoc*topicProbfromComment*topicProbInDoc;
                normalizedProb += m_topicProbCache[k];
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
            pDoc.m_sstat[tid] ++;
            word_topic_sstat[tid][wid] ++;
            m_sstat[tid]++;

        }
    }

    protected double parentWordByTopicProbFromCommentWordEmbed(int curRawPWId, int curPTId, int samplePTId, _ParentDoc4WordEmbedding pDoc){
        double wordTopicProb = 1.0;

        for(int rawWId=0; rawWId<m_rawFeatureSize; rawWId++){
            double widNum4SamplePTId = pDoc.m_commentThread_wordSS[1][samplePTId][rawWId];
            for (int i = 0; i < widNum4SamplePTId; i++) {
                wordTopicProb *= wordByTopicEmbedInComm(curRawPWId, curPTId, rawWId, samplePTId, samplePTId, pDoc);
                wordTopicProb /= wordByTopicEmbedInComm(curRawPWId, curPTId, rawWId, samplePTId, 0, pDoc);
            }

            double widNum4Sample0 = pDoc.m_commentThread_wordSS[1][0][rawWId];
            for (int i = 0; i < widNum4Sample0; i++) {
                wordTopicProb /= wordByTopicEmbedInComm(curRawPWId, curPTId, rawWId, 0, 0, pDoc);
                wordTopicProb *= wordByTopicEmbedInComm(curRawPWId, curPTId, rawWId, 0, samplePTId, pDoc);
            }

        }
        return wordTopicProb;
    }

    protected double wordByTopicEmbedInComm(int curRawPWId, int curPTId, int rawCWId, int cTId, int samplePTId, _ParentDoc pDoc){
        double wordEmbeddingSim = 0.0;

        double normalizedTerm = 0.0;

        for(_Word pWord:pDoc.getWords()){
            int pWId = pWord.getIndex();
            int pTId = pWord.getTopic();
            int rawPWId = pWord.getRawIndex();

            if(pTId!=cTId)
                continue;

            double wordCosSim = m_wordSimMatrix[rawCWId][rawPWId];
            wordEmbeddingSim += wordCosSim;
            normalizedTerm += m_wordSimVec[rawPWId];
        }

        if(curPTId == cTId){
            wordEmbeddingSim -= m_wordSimMatrix[rawCWId][curRawPWId];
            normalizedTerm -= m_wordSimVec[curRawPWId];
        }

        if(samplePTId == cTId){
            wordEmbeddingSim += m_wordSimMatrix[rawCWId][curRawPWId];
            normalizedTerm += m_wordSimVec[curRawPWId];
        }

        if(wordEmbeddingSim==0.0){
//            System.out.println("zero similarity for topic sampling parent\t"+cTId);
        }

        wordEmbeddingSim += d_beta;
        normalizedTerm += d_beta*vocabulary_size;
        wordEmbeddingSim /= normalizedTerm;
        return wordEmbeddingSim;
    }

    protected void sampleInChildDoc(_Doc d){
        int wid, tid, xid, rawWId;
        _ChildDoc cDoc = (_ChildDoc)d;
        _ParentDoc4WordEmbedding pDoc = (_ParentDoc4WordEmbedding)(cDoc.m_parentDoc);

        for(_Word w:cDoc.getWords()){
            double normalizedProb = 0.0;
            wid = w.getIndex();
            tid = w.getTopic();
            rawWId = w.getRawIndex();
//            xid = 0;
            xid = w.getX();

            cDoc.m_sstat[tid]--;
            if(xid==1)
                pDoc.m_commentThread_wordSS[xid][tid][rawWId]--;

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
                    wordTopicProb = wordByTopicEmbedInComm(rawWId, tid, pDoc);
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

            if(xid==1)
                pDoc.m_commentThread_wordSS[xid][tid][rawWId]++;
            if(xid==0) {
                if (m_collectCorpusStats) {
                    word_topic_sstat[tid][wid]++;
                    m_sstat[tid] ++;
                }
            }
        }
    }

    protected double wordByTopicEmbedInComm(int rawCWId, int tid, _ParentDoc pDoc){
        double wordEmbeddingSim = 0.0;

        double normalizedTerm = 0.0;

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
