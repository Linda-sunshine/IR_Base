package topicmodels.correspondenceModels;

import structures.*;
import utils.Utils;

import java.io.*;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Arrays;
import java.util.HashMap;

/**
 * Created by jetcai1900 on 3/18/17.
 */
public class wordEmbeddingBasedCorrModel extends PriorCorrLDA {

    double[] m_gamma;
    double[][] m_wordSimMatrix;
    double[] m_wordSimVec;
    double[] m_xProbCache;

    public wordEmbeddingBasedCorrModel(int number_of_iteration, double converge, double beta, _Corpus c,
                                       double lambda, int number_of_topics, double alpha, double alpha_c,
                                       double[]gamma, double burnIn, int lag){
        super(number_of_iteration, converge, beta, c, lambda,
                number_of_topics, alpha, alpha_c, burnIn, lag);
        d_alpha_c = alpha_c;
        int gammaLength = gamma.length;
        m_gamma = new double[gammaLength];
        System.arraycopy(gamma, 0, m_gamma, 0, gammaLength);
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

    public void loadWordSim4Corpus(String wordSimFileName){
        double simThreshold = 1;
        if(wordSimFileName == null||wordSimFileName.isEmpty()){
            return;
        }

        try{
            if(m_wordSimMatrix==null) {
                m_wordSimMatrix = new double[vocabulary_size][vocabulary_size];
                m_wordSimVec = new double[vocabulary_size];
            }

            double maxSim = -2;
            double minSim = 2;

            for(int v=0; v<vocabulary_size; v++) {
                Arrays.fill(m_wordSimMatrix[v], 0);
                Arrays.fill(m_wordSimVec, 0);
            }

            String tmpTxt;
            String[] lineContainer;

            HashMap<String, Integer> featureNameIndex = new HashMap<String, Integer>();
            for(int i=0; i<m_corpus.getFeatureSize(); i++){
                featureNameIndex.put(m_corpus.getFeature(i), featureNameIndex.size());
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
                    int rowWId = featureNameIndex.get(featureList.get(lineIndex));

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
        for(int i=0; i<vocabulary_size; i++) {
            m_wordSimVec[i] = 0;
            for (int j = 0; j < vocabulary_size; j++) {
//                System.out.println("word \t"+m_corpus.getFeature(i)+"\t j\t"+m_corpus.getFeature(j)+"\t"+m_wordSimMatrix[i][j]+"after");
                double normalizedSim = (m_wordSimMatrix[i][j] - minSim) / (maxSim - minSim);

                if(normalizedSim< threshold)
                    m_wordSimMatrix[i][j] = 0;
                else
                    m_wordSimMatrix[i][j] = normalizedSim;
                m_wordSimVec[i] += normalizedSim;
//                System.out.println("word \t"+m_corpus.getFeature(i)+"\t j\t"+m_corpus.getFeature(j)+"\t"+normalizedSim);
            }
        }
    }

    public void LoadPrior(String fileName, double eta) {
        if (fileName == null || fileName.isEmpty()) {
            return;
        }

        try{

            if (word_topic_prior == null) {
                word_topic_prior = new double[number_of_topics][vocabulary_size];
            }

            for (int k = 0; k < number_of_topics; k++)
                Arrays.fill(word_topic_prior[k], 0);

            String tmpTxt;
            String[] lineContainer;
            String[] featureContainer;
            int tid = 0;

            HashMap<String, Integer> featureNameIndex = new HashMap<String, Integer>();
            for(int i=0; i<m_corpus.getFeatureSize(); i++){
                featureNameIndex.put(m_corpus.getFeature(i), featureNameIndex.size());
            }

            BufferedReader br = new BufferedReader(new InputStreamReader(
                    new FileInputStream(fileName), "UTF-8"));

            while ((tmpTxt = br.readLine()) != null) {
                tmpTxt = tmpTxt.trim();
                if (tmpTxt.isEmpty())
                    continue;

                lineContainer = tmpTxt.split("\t");

                tid = Integer.parseInt(lineContainer[0]);
                for (int i = 1; i < lineContainer.length; i++) {
                    featureContainer = lineContainer[i].split(":");

                    String featureName = featureContainer[0];
                    double featureProb = Double.parseDouble(featureContainer[1]);

                    int featureIndex = featureNameIndex.get(featureName);

                    word_topic_prior[tid][featureIndex] = featureProb;
                }
            }

            System.out.println("prior is added");

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    public String toString(){
        return String.format("word embedding based Corr model [k:%d, alpha:%.2f, beta:%.2f, Gibbs Sampling]",
                number_of_topics, d_alpha, d_beta);
    }

    public void EM() {
        System.out.format("Starting %s...\n", toString());

        long starttime = System.currentTimeMillis();

        m_collectCorpusStats = true;
        initialize_probability(m_trainSet);

        double delta = 0, last = 0, current = 0;
        int i = 0, displayCount = 0;
        do {

            long eStartTime = System.currentTimeMillis();

            init();
            for(_Doc d:m_trainSet) {
                calculate_E_step(d);
                sampleX4Child(d);
                sanityCheck(d);
            }

            long eEndTime = System.currentTimeMillis();

            System.out.println("per iteration e step time\t"
                    + (eEndTime - eStartTime) / 1000.0 + "\t seconds");

            long mStartTime = System.currentTimeMillis();
            calculate_M_step(i);
            long mEndTime = System.currentTimeMillis();

            System.out.println("per iteration m step time\t"
                    + (mEndTime - mStartTime) / 1000.0 + "\t seconds");

//            if (m_converge > 0
//                    || (m_displayLap > 0 && i % m_displayLap == 0 && displayCount > 6)) {
//                // required to display log-likelihood
//                current = calculate_log_likelihood();
//                // together with corpus-level log-likelihood
//
//                if (i > 0)
//                    delta = (last - current) / last;
//                else
//                    delta = 1.0;
//                last = current;
//            }

//            if (m_displayLap > 0 && i % m_displayLap == 0) {
//                if (m_converge > 0) {
//                    System.out.format(
//                            "Likelihood %.3f at step %s converge to %f...\n",
//                            current, i, delta);
//                    infoWriter.format(
//                            "Likelihood %.3f at step %s converge to %f...\n",
//                            current, i, delta);
//
//                } else {
//                    System.out.print(".");
//                    if (displayCount > 6) {
//                        System.out.format("\t%d:%.3f\n", i, current);
//                        infoWriter.format("\t%d:%.3f\n", i, current);
//                    }
//                    displayCount++;
//                }
//            }

//            if (m_converge > 0 && Math.abs(delta) < m_converge)
//                break;// to speed-up, we don't need to compute likelihood in
            // many cases
        } while (++i < this.number_of_iteration);

        finalEst();

//        long endtime = System.currentTimeMillis() - starttime;
//        System.out.format("Likelihood %.3f after step %s converge to %f after %d seconds...\n", current, i, delta, endtime / 1000);
//        infoWriter.format("Likelihood %.3f after step %s converge to %f after %d seconds...\n", current, i, delta, endtime / 1000);
    }

    protected void sanityCheck(_Doc d){
        if(d instanceof _ParentDoc){
            double cWordNumInSS = 0.0;
            _ParentDoc4WordEmbedding pDoc = (_ParentDoc4WordEmbedding)d;

            for(int x=0; x<m_gamma.length; x++){
                for(int k=0; k<number_of_topics; k++){
                    for(int v=0; v<vocabulary_size; v++){
                        cWordNumInSS += pDoc.m_commentThread_wordSS[x][k][v];
                    }
                }
            }

            double cWordNum = 0.0;
            for(_ChildDoc cDoc:pDoc.m_childDocs){
                cWordNum += cDoc.getTotalDocLength();
            }

            if(cWordNum==cWordNumInSS){
//                System.out.println("pass sanityCheck");
            }else{
//                System.out.println("Wuwu, wrong sanityCheck");
            }
        }
    }

    @Override
    protected void sampleInParentDoc(_Doc d){
        int wid, tid, xid;

        _ParentDoc4WordEmbedding pDoc = (_ParentDoc4WordEmbedding)d;

//        for(tid=0; tid< number_of_topics; tid++) {
//            for (wid = 0; wid < vocabulary_size; wid++) {
//                word_topic_sstat[tid][wid] -= pDoc.m_commentThread_wordSS[0][tid][wid];
//                m_sstat[tid] -= pDoc.m_commentThread_wordSS[0][tid][wid];
//            }
//        }

        for(_Word w:pDoc.getWords()){
            double normalizedProb = 0;

            wid = w.getIndex();
            tid = w.getTopic();

            pDoc.m_sstat[tid] --;
            word_topic_sstat[tid][wid]--;
            m_sstat[tid] --;


            for(int k=0; k<number_of_topics; k++){
                double wordTopicProbfromCommentWordEmbed = parentWordByTopicProbFromCommentWordEmbed(wid, tid, k, pDoc);
//                double wordTopicProbfromCommentPhi = parentWordTopicProbfromCommentPhi(k, pDoc);
                double wordTopicProbInDoc = parentWordByTopicProb(k, wid)/parentWordByTopicProb(0, wid);
                double topicProbfromComment = parentChildInfluenceProb(k, pDoc);
                double topicProbInDoc = parentTopicInDocProb(k, pDoc);

                m_topicProbCache[k] = wordTopicProbfromCommentWordEmbed
                        *wordTopicProbInDoc*topicProbfromComment*topicProbInDoc;

//                m_topicProbCache[k] = wordTopicProbInDoc*topicProbfromComment*topicProbInDoc;
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

//        for(tid=0; tid< number_of_topics; tid++) {
//            for (wid = 0; wid < vocabulary_size; wid++) {
//                word_topic_sstat[tid][wid] += pDoc.m_commentThread_wordSS[0][tid][wid];
//                m_sstat[tid] += pDoc.m_commentThread_wordSS[0][tid][wid];
//            }
//        }

    }

    protected double parentWordByTopicProbFromCommentWordEmbed(int curPWId, int curPTId, int samplePTId, _ParentDoc4WordEmbedding pDoc){
        double wordTopicProb = 1.0;

        for(int wid=0; wid<vocabulary_size; wid++){
            double widNum4SamplePTId = pDoc.m_commentThread_wordSS[1][samplePTId][wid];
            for (int i = 0; i < widNum4SamplePTId; i++) {
                wordTopicProb *= wordByTopicEmbedInComm(curPWId, curPTId, wid, samplePTId, samplePTId, pDoc);
                wordTopicProb /= wordByTopicEmbedInComm(curPWId, curPTId, wid, samplePTId, 0, pDoc);
            }

            double widNum4Sample0 = pDoc.m_commentThread_wordSS[1][0][wid];
            for (int i = 0; i < widNum4Sample0; i++) {
                wordTopicProb /= wordByTopicEmbedInComm(curPWId, curPTId, wid, 0, 0, pDoc);
                wordTopicProb *= wordByTopicEmbedInComm(curPWId, curPTId, wid, 0, samplePTId, pDoc);
            }

        }
        return wordTopicProb;
    }

    protected double parentWordByTopicProb(int tid, int wid){
        double wordTopicProb = 0;

        wordTopicProb = word_topic_sstat[tid][wid]/m_sstat[tid];

        return wordTopicProb;
    }

    @Override
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
                    wordTopicProb = wordByTopicEmbedInComm(wid, tid, pDoc);
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

    ////x=0, phi; x=1, wordembedding
    protected void sampleX4Child(_Doc d){
        if(d instanceof _ParentDoc)
            return;

        _ChildDoc cDoc = (_ChildDoc)d;
        _ParentDoc4WordEmbedding pDoc = (_ParentDoc4WordEmbedding)(cDoc.m_parentDoc);

        for(_Word w:cDoc.getWords()){
            double normalizedProb = 0;
            int wid = w.getIndex();
            int tid = w.getTopic();
            int xid = w.getX();

//            cDoc.m_xTopicSstat[xid][tid]--;
            cDoc.m_xSstat[xid]--;
            pDoc.m_commentThread_wordSS[xid][tid][wid] --;

            if(xid==0) {
                word_topic_sstat[tid][wid]--;
                m_sstat[tid]--;
            }

            double wordEmbeddingSim = wordByTopicEmbedInComm(wid, tid, pDoc);
            double wordTopicProb =  wordByTopicProbInComm(wid, tid);
            double x0Prob = xProbInComm(0, cDoc);
            double x1Prob = xProbInComm(1, cDoc);

            m_xProbCache[0] = wordTopicProb*x0Prob;
            m_xProbCache[1] = wordEmbeddingSim*x1Prob;

            normalizedProb += m_xProbCache[0];
            normalizedProb += m_xProbCache[1];

            normalizedProb *= m_rand.nextDouble();
            if(normalizedProb<=m_xProbCache[0])
                xid = 0;
            else
                xid = 1;

            w.setX(xid);
            cDoc.m_xSstat[xid]++;
            pDoc.m_commentThread_wordSS[xid][tid][wid]++;

            if(xid==0) {
                word_topic_sstat[tid][wid]++;
                m_sstat[tid]++;
            }
        }
    }

    //this one is specially designed for the change of the topic of the word in the article
    //cTId is the topic to be assigned to the word in the article
    protected double wordByTopicEmbedInComm(int curPWId, int curPTId, int cWId, int cTId, int samplePTId, _ParentDoc pDoc){
        double wordEmbeddingSim = 0.0;

        double normalizedTerm = 0.0;

        for(_Word pWord:pDoc.getWords()){
            int pWId = pWord.getIndex();
            int pTId = pWord.getTopic();

            if(pTId!=cTId)
                continue;

            double wordCosSim = m_wordSimMatrix[cWId][pWId];
            wordEmbeddingSim += wordCosSim;
            normalizedTerm += m_wordSimVec[pWId];
        }

        if(curPTId != cTId){
            wordEmbeddingSim -= m_wordSimMatrix[cWId][curPWId];
            normalizedTerm -= m_wordSimVec[curPWId];
        }

        if(samplePTId == cTId){
            wordEmbeddingSim += m_wordSimMatrix[cWId][curPWId];
            normalizedTerm += m_wordSimVec[curPWId];
        }

        if(wordEmbeddingSim==0.0){
//            System.out.println("zero similarity for topic sampling parent\t"+cTId);
        }

        wordEmbeddingSim += d_beta;
        normalizedTerm += d_beta*vocabulary_size;
        wordEmbeddingSim /= normalizedTerm;
        return wordEmbeddingSim;
    }



//
//    protected double wordByTopicEmbedInComm(int curPWId, int curPTId, int cWId, int cTId, _ParentDoc pDoc){
//        double wordEmbeddingSim = 0.0;
//
//        double normalizedTerm = 0.0;
//
//        for (_Word pWord : pDoc.getWords()) {
//            int pWId = pWord.getIndex();
//            int pTId = pWord.getTopic();
//
//            if (pTId != cTId)
//                continue;
//
//            if (pWId == cWId) {
//                wordEmbeddingSim += 1;
//            }
//
//        }
//
//        if(curPTId != cTId){
//            if(curPWId==cWId)
//                wordEmbeddingSim += 1;
//            normalizedTerm += 1;
//        }
//
//        normalizedTerm += pDoc.m_sstat[cTId];
//        wordEmbeddingSim += d_beta;
//        normalizedTerm += d_beta*vocabulary_size;
//
//        wordEmbeddingSim /= normalizedTerm;
//        return wordEmbeddingSim;
//    }

    protected double wordByTopicEmbedInComm(int wid, int tid, _ParentDoc pDoc){
        double wordEmbeddingSim = 0.0;

        double normalizedTerm = 0.0;

        for(_Word pWord:pDoc.getWords()){
            int pWId = pWord.getIndex();
            int pTId = pWord.getTopic();

            if(pTId!=tid)
                continue;

            double wordCosSim = m_wordSimMatrix[wid][pWId];

            wordEmbeddingSim += wordCosSim;
            normalizedTerm += m_wordSimVec[pWId];
        }

        if(wordEmbeddingSim==0.0){
//            System.out.println("zero similarity for topic child\t"+tid);
        }

        wordEmbeddingSim += d_beta;
        normalizedTerm += d_beta*vocabulary_size;

        wordEmbeddingSim /= normalizedTerm;
        return wordEmbeddingSim;
    }

//    protected double wordByTopicEmbedInComm(int wid, int tid, _ParentDoc pDoc){
//        double wordEmbeddingSim = 0.0;
//
//        double normalizedTerm = 0.0;
//
//        for (_Word pWord : pDoc.getWords()) {
//            int pWId = pWord.getIndex();
//            int pTId = pWord.getTopic();
//
//            if (pTId != tid)
//                continue;
//
//            if(pWId == wid){
//                wordEmbeddingSim += 1;
//            }
//
//        }
//
//        wordEmbeddingSim += d_beta;
//        normalizedTerm = pDoc.m_sstat[tid]+d_beta*vocabulary_size;
//
//        wordEmbeddingSim /= normalizedTerm;
//        return wordEmbeddingSim;
//    }

    protected double wordByTopicProbInComm(int wid, int tid){
        double wordTopicProb = 0.0;

        wordTopicProb = word_topic_sstat[tid][wid]/m_sstat[tid];

        return wordTopicProb;
    }

    protected double xProbInComm(int xid, _ChildDoc cDoc){
        return cDoc.m_xSstat[xid]+m_gamma[xid];
    }

    @Override
    protected void collectStats(_Doc d) {
        if(d instanceof  _ParentDoc){
            for(int k=0; k<this.number_of_topics; k++)
                d.m_topics[k] += d.m_sstat[k]+d_alpha;
        }

        if(d instanceof _ChildDoc){
            _ChildDoc cDoc = (_ChildDoc)d;
            for(int k=0; k<number_of_topics; k++)
                cDoc.m_topics[k] += cDoc.m_sstat[k]+m_alpha_c[k];
//
            for(int x=0; x<m_gamma.length; x++)
                cDoc.m_xProportion[x] += cDoc.m_xSstat[x]+m_gamma[x];

            for(_Word w:d.getWords()){
                w.collectXStats();
            }
        }
    }

    protected double calculate_log_likelihood4Parent(_Doc d){
        _ParentDoc pDoc = (_ParentDoc)d;
        double docLogLikelihood = 0;

        _SparseFeature[] fv = pDoc.getSparse();
        return 0;
    }

    protected double calculate_log_likelihood4Child(_Doc d) {
        return 0;
    }

    @Override
    protected void estThetaInDoc(_Doc d){
        if(d instanceof _ParentDoc){
            Utils.L1Normalization(d.m_topics);
        }else{
            _ChildDoc cDoc = (_ChildDoc)d;
            Utils.L1Normalization(cDoc.m_topics);
            Utils.L1Normalization(cDoc.m_xProportion);

            for(_Word w:d.getWords()){
                w.getXProb();
            }
        }
    }


}