package topicmodels.correspondenceModels;

import structures.*;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;

/**
 * Created by jetcai1900 on 4/27/17.
 */
public class BitermTopicModel4AC extends BitermTopicModel{

    double[][] m_wordSimMatrix;
    double[] m_wordSimVec;
    int m_rawFeatureSize;

    public BitermTopicModel4AC(int number_of_iteration, double converge, double beta,
                               _Corpus c, double lambda,
                               int number_of_topics, double alpha, double burnIn, int lag){
        super(number_of_iteration, converge, beta, c, lambda, number_of_topics, alpha, burnIn, lag);

        m_rawFeatureSize = c.m_rawFeatureList.size();
    }

    @Override
    public String toString() {
        return String.format("BitermTopicModel4AC[k:%d, alpha:%.2f, beta:%.2f, trainProportion:%.2f, Gibbs Sampling]", number_of_topics, d_alpha, d_beta, 1-m_testWord4PerplexityProportion);
    }

    public void loadWordSim4Corpus(String wordSimFileName){
        double simThreshold = -1;
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

    //construct biterms
    protected void constructBiterms(_Doc d){
        if(d instanceof  _ParentDoc) {
            return;
        }
//        _DocWithRawToken doc = (_DocWithRawToken)d;
        _ChildDoc4BitermTM cDoc = (_ChildDoc4BitermTM)d;

        _ParentDoc pDoc = cDoc.m_parentDoc;
        // whether we can move this to initialization
//        cDoc.createWords4TM(number_of_topics, d_alpha);
        cDoc.createWords4TM(number_of_topics, d_alpha);

        for(int i=0; i<d.getTotalDocLength(); i++){
            for(int j=i+1; j<d.getTotalDocLength(); j++){
                int wid1 = d.getWordByIndex(i).getIndex();
                int wid2 = d.getWordByIndex(j).getIndex();
                Biterm bitermObj = new Biterm(wid1, wid2);
                bitermObj.setTopics4GibbsbyRawToken(number_of_topics);
                m_bitermList.add(bitermObj);
                cDoc.addBiterm(bitermObj);
            }
        }

        for(int i=0; i<cDoc.getTotalDocLength(); i++){
            int wid1 = cDoc.getWordByIndex(i).getIndex();
            int maxWid2 = -1;
            double maxSim = -1;

            for(_Word w:pDoc.getWords()){
                int wid2 = w.getIndex();
                if(wid2==wid1)
                    continue;
                double sim = m_wordSimMatrix[wid1][wid2];
                if(sim > maxSim){
                    maxSim = sim;
                    maxWid2 = wid2;
                }
            }

            if(maxSim < 0.95)
                continue;
//
//            if(maxSim < 0.85){
//                maxWid2 = pDoc.getWords()[m_rand.nextInt(pDoc.getTotalDocLength())].getIndex();
//            }
//
//            if(maxSim < 1){
//                maxWid2 = cDoc.getWords()[i+m_rand.nextInt(cDoc.getTotalDocLength()-i)].getIndex();
//            }

            Biterm bitermObj = new Biterm(wid1, maxWid2);
            bitermObj.setTopics4GibbsbyRawToken(number_of_topics);
            m_bitermList.add(bitermObj);
            cDoc.addBiterm(bitermObj);
        }

    }

}
