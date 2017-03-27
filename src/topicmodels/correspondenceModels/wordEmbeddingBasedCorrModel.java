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
                pDoc.setTopics4Gibbs(number_of_topics, 0, vocabulary_size, m_gamma.length);

                for(_ChildDoc cDoc:pDoc.m_childDocs) {
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

                parentWordByTopicFromCommentWordEmbedSS(pDoc);
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
        double simThreshold = -1;
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
                double normalizedSim = (m_wordSimMatrix[i][j] - minSim) / (maxSim - minSim);

                if(normalizedSim< threshold)
                    m_wordSimMatrix[i][j] = 0;
                else
                    m_wordSimMatrix[i][j] = normalizedSim;
                    m_wordSimVec[i] += normalizedSim;
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
            }

            long eEndTime = System.currentTimeMillis();

            System.out.println("per iteration e step time\t"
                    + (eEndTime - eStartTime) / 1000.0 + "\t seconds");

            long mStartTime = System.currentTimeMillis();
            calculate_M_step(i);
            long mEndTime = System.currentTimeMillis();

            System.out.println("per iteration m step time\t"
                    + (mEndTime - mStartTime) / 1000.0 + "\t seconds");

        } while (++i < this.number_of_iteration);

        finalEst();

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

        for(_Word w:pDoc.getWords()){
            double normalizedProb = 0;

            wid = w.getIndex();
            tid = w.getTopic();

            pDoc.m_sstat[tid] --;
            word_topic_sstat[tid][wid]--;
            m_sstat[tid] --;

            removeWordSimFromCommentWordEmbedSS(wid, tid, pDoc);

            double wordTopicProbInDoc4Zero = parentWordByTopicProb(0, wid);
            for(int k=0; k<number_of_topics; k++){
                addWordSimFromCommentWordEmbedSS(wid, k, pDoc);
                double wordTopicProbfromCommentWordEmbed = parentWordByTopicProbFromCommentWordEmbed(k, pDoc);
                double wordTopicProbInDoc = parentWordByTopicProb(k, wid)/wordTopicProbInDoc4Zero;
                double topicProbfromComment = parentChildInfluenceProb(k, pDoc);
                double topicProbInDoc = parentTopicInDocProb(k, pDoc);

                m_topicProbCache[k] = wordTopicProbfromCommentWordEmbed
                        *wordTopicProbInDoc*topicProbfromComment*topicProbInDoc;
                normalizedProb += m_topicProbCache[k];

                removeWordSimFromCommentWordEmbedSS(wid, k, pDoc);
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

            addWordSimFromCommentWordEmbedSS(wid, tid, pDoc);
        }

    }

    //store all information and directly call when computing the probability
    protected void parentWordByTopicFromCommentWordEmbedSS(_ParentDoc4WordEmbedding pDoc){
        for(int k=0; k<number_of_topics; k++){
            for(int v=0; v<vocabulary_size; v++){
                Arrays.fill(pDoc.m_commentWordEmbed_simSS[k][v], 0.0);
            }
            Arrays.fill(pDoc.m_commentWordEmbed_normSimSS[k], 0.0);
        }

        for(_Word w: pDoc.getWords()){
            int wid = w.getIndex();
            int tid = w.getTopic();

            for(int v=0; v<vocabulary_size; v++) {
                for(int k=0; k<number_of_topics; k++) {
                    if(k==tid)
                        pDoc.m_commentWordEmbed_simSS[tid][v][0] += m_wordSimMatrix[wid][v];
                    else
                        pDoc.m_commentWordEmbed_simSS[k][v][1] += 1 - m_wordSimMatrix[wid][v];
                }
            }

            for(int k=0; k<number_of_topics; k++) {
                if(k==tid)
                    pDoc.m_commentWordEmbed_normSimSS[tid][0] += m_wordSimVec[wid];
                else
                    pDoc.m_commentWordEmbed_normSimSS[k][1] += vocabulary_size- m_wordSimVec[wid];
            }


        }

    }

    protected void removeWordSimFromCommentWordEmbedSS(int curPWId, int curPTId, _ParentDoc4WordEmbedding pDoc){

        for (int v = 0; v < vocabulary_size; v++){
            for(int k=0; k<number_of_topics; k++) {
                if(k==curPTId) {
                    pDoc.m_commentWordEmbed_simSS[curPTId][v][0] -= m_wordSimMatrix[curPWId][v];
                    pDoc.m_commentWordEmbed_normSimSS[curPTId][0] -= m_wordSimVec[curPWId];
                }
                else {
                    pDoc.m_commentWordEmbed_simSS[k][v][1] -= (1-m_wordSimMatrix[curPWId][v]);
                    pDoc.m_commentWordEmbed_normSimSS[k][1] -= (vocabulary_size-m_wordSimVec[curPWId]);
                }
            }
        }

    }

    protected void addWordSimFromCommentWordEmbedSS(int curPWId, int samplePTId, _ParentDoc4WordEmbedding pDoc){

        for (int v = 0; v < vocabulary_size; v++){
            for(int k=0; k<number_of_topics; k++) {
                if(k==samplePTId) {
                    pDoc.m_commentWordEmbed_simSS[samplePTId][v][0] += m_wordSimMatrix[curPWId][v];
                    pDoc.m_commentWordEmbed_normSimSS[samplePTId][0] += m_wordSimVec[curPWId];
                }
                else {
                    pDoc.m_commentWordEmbed_simSS[k][v][1] += (1-m_wordSimMatrix[curPWId][v]);
                    pDoc.m_commentWordEmbed_normSimSS[k][1] += (vocabulary_size-m_wordSimVec[curPWId]);
                }
            }
        }

    }

    protected double parentWordByTopicProbFromCommentWordEmbed(int curPWId, int samplePTId, _ParentDoc4WordEmbedding pDoc) {
        double wordTopicProb = 1.0;

        for(int k=0; k<number_of_topics; k++){
            for(int wid=0; wid<vocabulary_size; wid++){
                double widNum4K = pDoc.m_commentThread_wordSS[1][k][wid];
                if(widNum4K==0)
                    continue;

                double wordNum4KinArticle = pDoc.m_sstat[k];
                double wordNum4ZeroinArticle = pDoc.m_sstat[0];

                if(k==samplePTId)
                    wordNum4KinArticle ++;
                if(k==0)
                    wordNum4ZeroinArticle ++;

                double commentWordTopicProb4SampleSim = pDoc.m_commentWordEmbed_simSS[k][wid][0];
                double commentWordTopicProb4SampleNormTermSim = pDoc.m_commentWordEmbed_normSimSS[k][0];
                if(k==samplePTId) {
                    commentWordTopicProb4SampleSim += m_wordSimMatrix[curPWId][wid];
                    commentWordTopicProb4SampleNormTermSim += m_wordSimVec[curPWId];
                }

                commentWordTopicProb4SampleSim /= wordNum4KinArticle;
                commentWordTopicProb4SampleNormTermSim /= wordNum4KinArticle;

                double commentWordTopicProb4SampleDissim = pDoc.m_commentWordEmbed_simSS[k][wid][1];
                double commentWordTopicProb4SampleNormTermDissim = pDoc.m_commentWordEmbed_normSimSS[k][1]/(pDoc.getTotalDocLength()-wordNum4KinArticle);
                if(k!=samplePTId) {
                    commentWordTopicProb4SampleDissim += 1 - m_wordSimMatrix[curPWId][wid];
                    commentWordTopicProb4SampleNormTermDissim += (vocabulary_size-m_wordSimVec[curPWId]);
                }
                commentWordTopicProb4SampleDissim /= (pDoc.getTotalDocLength()-wordNum4KinArticle);
                commentWordTopicProb4SampleNormTermDissim /= (pDoc.getTotalDocLength()-wordNum4KinArticle);

                wordTopicProb *= (commentWordTopicProb4SampleSim+commentWordTopicProb4SampleDissim)/(commentWordTopicProb4SampleNormTermSim+commentWordTopicProb4SampleNormTermDissim);

//                double commentWordTopicProb4Zero = pDoc.m_commentWordEmbed_simSS[0][wid][0]/wordNum4ZeroinArticle;
//                commentWordTopicProb4Zero += pDoc.m_commentWordEmbed_simSS[0][wid][1]/(pDoc.getTotalDocLength()-wordNum4ZeroinArticle);
//
//                double commentWordTopicProb4ZeroNormTerm = pDoc.m_commentWordEmbed_normSimSS[0][0]/wordNum4ZeroinArticle;
//                commentWordTopicProb4ZeroNormTerm += pDoc.m_commentWordEmbed_normSimSS[0][1]/(pDoc.getTotalDocLength()-wordNum4ZeroinArticle);
//
//                wordTopicProb *= commentWordTopicProb4Sample/commentWordTopicProb4Zero;
//                wordTopicProb *= commentWordTopicProb4ZeroNormTerm/commentWordTopicProb4SampleNormTerm;
            }
        }

        return wordTopicProb;
    }

    protected double parentWordByTopicProbFromCommentWordEmbed(int curPWId, int curPTId, int samplePTId, _ParentDoc4WordEmbedding pDoc){
        double wordTopicProb = 1.0;

        if(samplePTId==0)
            return wordTopicProb;

        for(int k=0; k<number_of_topics; k++) {
            for (int wid = 0; wid < vocabulary_size; wid++) {
                double widNum4K = pDoc.m_commentThread_wordSS[1][k][wid];
                if(widNum4K==0.0)
                    continue;
                double commentWordTopicProb4Sample = wordByTopicEmbedInComm(curPWId, curPTId, wid, k, samplePTId, pDoc);
                double commentWordTopicProb40 = wordByTopicEmbedInComm(curPWId, curPTId, wid, k, 0, pDoc);
                double commentWordTopicProb = commentWordTopicProb4Sample/commentWordTopicProb40;

                for (int i = 0; i < widNum4K; i++) {
                    wordTopicProb *= commentWordTopicProb;
                }
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

        double wordSim = 0;
        double wordDissim = 0;

        double normSimTerm = 0;
        double normDissimTerm = 0;

        double simTopicNum = pDoc.m_sstat[cTId];
        double dissimTopicNum = pDoc.getTotalDocLength()-pDoc.m_sstat[cTId]-1;

        for(_Word pWord:pDoc.getWords()){
            int pWId = pWord.getIndex();
            int pTId = pWord.getTopic();

            if(pTId!=cTId){
                double wordCosSim = 1-m_wordSimMatrix[cWId][pWId];
                wordDissim += wordCosSim;
                normDissimTerm += vocabulary_size-m_wordSimVec[pWId];
            }else {
                double wordCosSim = m_wordSimMatrix[cWId][pWId];
                wordSim += wordCosSim;
                normSimTerm += m_wordSimVec[pWId];
            }
        }

        if(curPTId == cTId){
            wordSim -= m_wordSimMatrix[cWId][curPWId];
            normSimTerm -= m_wordSimVec[curPWId];
        }else{
            wordDissim -= (1-m_wordSimMatrix[cWId][curPWId]);
            normDissimTerm -= (vocabulary_size-m_wordSimVec[curPWId]);
        }

        if(samplePTId == cTId){
            wordSim += m_wordSimMatrix[cWId][curPWId];
            normSimTerm += m_wordSimVec[curPWId];

            simTopicNum ++;
        }else{
            wordDissim += 1-m_wordSimMatrix[cWId][curPWId];
            normDissimTerm += vocabulary_size-m_wordSimVec[curPWId];
            dissimTopicNum ++;
        }

        wordEmbeddingSim = wordSim/simTopicNum+wordDissim/dissimTopicNum;
        normalizedTerm = normSimTerm/simTopicNum+normDissimTerm/dissimTopicNum;

        if(wordEmbeddingSim==0.0){
//            System.out.println("zero similarity for topic sampling parent\t"+cTId);
        }

        wordEmbeddingSim += d_beta;
        normalizedTerm += d_beta*vocabulary_size;
        wordEmbeddingSim /= normalizedTerm;
        return wordEmbeddingSim;
    }

    protected double wordByTopicEmbedInComm(int wid, int tid, _ParentDoc4WordEmbedding pDoc){
        double wordTopicProb = 0.0;

        double wordSim = pDoc.m_commentWordEmbed_simSS[tid][wid][0];
        double wordDissim = pDoc.m_commentWordEmbed_simSS[tid][wid][1];

        double normSim = pDoc.m_commentWordEmbed_normSimSS[tid][0];
        double normDissim = pDoc.m_commentWordEmbed_normSimSS[tid][1];

        wordTopicProb = wordSim/pDoc.m_sstat[tid]+wordDissim/(pDoc.getTotalDocLength()-pDoc.m_sstat[tid]);
        wordTopicProb /= normSim/pDoc.m_sstat[tid]+normDissim/(pDoc.getTotalDocLength()-pDoc.m_sstat[tid]);

        return wordTopicProb;
    }

//    protected double wordByTopicEmbedInComm(int wid, int tid, _ParentDoc pDoc){
//        double wordEmbeddingSim = 0.0;
//        double normalizedTerm = 0.0;
//
//        double wordSim = 0;
//        double wordDissim = 0;
//
//        double normSimTerm = 0;
//        double normDissimTerm = 0;
//
//        for(_Word pWord:pDoc.getWords()){
//            int pWId = pWord.getIndex();
//            int pTId = pWord.getTopic();
//
//            if(pTId!=tid){
//                double wordCosSim = 1-m_wordSimMatrix[wid][pWId];
//                wordDissim += wordCosSim;
//                normDissimTerm += vocabulary_size-m_wordSimVec[pWId];
//            }else{
//                double wordCosSim = m_wordSimMatrix[wid][pWId];
//
//                wordSim += wordCosSim;
//                normSimTerm += m_wordSimVec[pWId];
//            }
//
//        }
//
//        wordEmbeddingSim = wordSim/pDoc.m_sstat[tid]+wordDissim/(pDoc.getTotalDocLength()-pDoc.m_sstat[tid]);
//        normalizedTerm = normSimTerm/pDoc.m_sstat[tid]+normDissimTerm/(pDoc.getTotalDocLength()-pDoc.m_sstat[tid]);
//
//        if(wordEmbeddingSim==0.0){
////            System.out.println("zero similarity for topic child\t"+tid);
//        }
//
//        wordEmbeddingSim += d_beta;
//        normalizedTerm += d_beta*vocabulary_size;
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
