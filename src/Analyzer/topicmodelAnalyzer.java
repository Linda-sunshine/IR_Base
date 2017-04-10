package Analyzer;

import net.didion.jwnl.data.Exc;
import opennlp.tools.util.InvalidFormatException;
import opennlp.tools.util.StringList;
import structures.*;
import utils.Utils;

import java.io.*;
import java.lang.reflect.Array;
import java.util.*;


/**
 * Created by jetcai1900 on 3/31/17.
 *
 * preprocess the documents used for topic model
 */
public class topicmodelAnalyzer extends ParentChildAnalyzer{
    class rawToken{
        String m_rawToken;
        double[] m_tokenEmbeddingVec;
        public rawToken(String rawTokenStr){
            m_rawToken = rawTokenStr;
        }

        public void setEmbeddingVec(double[] embeddingVec){
            m_tokenEmbeddingVec = new double[embeddingVec.length];
            System.arraycopy(embeddingVec, 0, m_tokenEmbeddingVec, 0, embeddingVec.length);
        }

        public double[] getEmbeddingVec(){
            return m_tokenEmbeddingVec;
        }
    }

    HashMap<String, rawToken> m_rawTokenMap;
    ArrayList<String> m_rawTokenStrList;
    ArrayList<String> m_outputFeatureList;
    ArrayList<String> m_rawFeatureList;
//    int embeddingSize = 100;

    public topicmodelAnalyzer(String tokenModel, int classNo, String providedCV, int Ngram, int threshold, String rawFeatureFile)
            throws InvalidFormatException, FileNotFoundException, IOException {
        super(tokenModel, classNo, providedCV, Ngram, threshold);
        parentHashMap = new HashMap<String, _ParentDoc>();
        m_rawTokenMap = new HashMap<String, rawToken>();
        m_rawTokenStrList = new ArrayList<String>();
        m_outputFeatureList = new ArrayList<String>();
        m_rawFeatureList = new ArrayList<String>();

        try{

            if (rawFeatureFile==null || rawFeatureFile.isEmpty())
                return;

            String tmpTxt;
            String[] lineContainer;

            BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(rawFeatureFile), "UTF-8"));
            while((tmpTxt=br.readLine())!=null){
                tmpTxt = tmpTxt.trim();
                if(tmpTxt.isEmpty())
                    continue;

                lineContainer = tmpTxt.split(" ");
                String rawWordStr = lineContainer[0];

                m_rawFeatureList.add(rawWordStr);

            }
        }catch(Exception e){
            System.err.format("[Error]Failed to open file %s!!", rawFeatureFile);
            return;
        }

    }

    protected boolean AnalyzeDoc(_Doc doc) {
        TokenizeResult result = TokenizerNormalizeStemmer(doc.getSource());// Three-step analysis.
        String[] tokens = result.getTokens();
        String[] rawTokens = result.getRawTokens();

        int y = doc.getYLabel();

        ArrayList<_Word> wordList = new ArrayList<_Word>();
        HashMap<Integer, Double> spVct = constructSpVct(tokens, rawTokens, y, null, wordList);
//        HashMap<Integer, Double> spVct = constructSpVct(rawTokens, rawTokens, y, null, wordList);

        if (spVct.size()>m_lengthThreshold) {
            doc.createSpVct(spVct);
            doc.setStopwordProportion(result.getStopwordProportion());
            doc.setWords(wordList);

            for(_Word w:doc.getWords()){
                String rawToken = w.getRawToken();
                if(!m_isCVLoaded){
                    if(!m_rawTokenStrList.contains(rawToken)){
                        m_rawTokenStrList.add(rawToken);
                    }
                }else{
                    if(!m_rawFeatureList.contains(rawToken)){
                        System.out.println("error raw feature\t"+rawToken);
                    }
                }
            }


            m_corpus.addDoc(doc);
            m_classMemberNo[y]++;
            if (m_releaseContent)
                doc.clearSource();
            return true;
        } else {
            /****Roll back here!!******/
            rollBack(spVct, y);
            return false;
        }
    }

    protected HashMap<Integer, Double> constructSpVct(String[] tokens, String[] rawTokens, int y, HashMap<Integer, Double> docWordMap, ArrayList<_Word> wordList) {
        int index = 0;
        double value = 0;
        HashMap<Integer, Double> spVct = new HashMap<Integer, Double>(); // Collect the index and counts of features.

        for(int tokenIndex=0; tokenIndex<tokens.length; tokenIndex++){
            String token = tokens[tokenIndex];
            String rawToken = rawTokens[tokenIndex];
            if (!m_isCVLoaded) {
                if (m_featureNameIndex.containsKey(token)) {
                    index = m_featureNameIndex.get(token);
                    if (spVct.containsKey(index)) {
                        value = spVct.get(index) + 1;
                        spVct.put(index, value);
                    } else {
                        spVct.put(index, 1.0);
                        if (docWordMap==null || !docWordMap.containsKey(index)) {
                            if(m_featureStat.containsKey(token))
                                m_featureStat.get(token).addOneDF(y);
                        }
                    }
                } else {// indicate we allow the analyzer to dynamically expand the feature vocabulary
                    expandVocabulary(token);// update the m_featureNames.
                    index = m_featureNameIndex.get(token);
                    spVct.put(index, 1.0);
                    if(m_featureStat.containsKey(token))
                        m_featureStat.get(token).addOneDF(y);
                }
                if(m_featureStat.containsKey(token))
                    m_featureStat.get(token).addOneTTF(y);

                _Word w = new _Word(index);
                rawToken = Normalize(rawToken);
                w.setRawToken(rawToken);
                wordList.add(w);

            } else if (m_featureNameIndex.containsKey(token)) {// CV is loaded.
                rawToken = Normalize(rawToken);
                if(!m_rawFeatureList.contains(rawToken))
                    continue;
                index = m_featureNameIndex.get(token);
                if (spVct.containsKey(index)) {
                    value = spVct.get(index) + 1;
                    spVct.put(index, value);
                } else {
                    spVct.put(index, 1.0);
                    if (!m_isCVStatLoaded)
                        m_featureStat.get(token).addOneDF(y);
                }

                if (!m_isCVStatLoaded)
                    m_featureStat.get(token).addOneTTF(y);

                _Word w = new _Word(index);

                int rawIndex = m_rawFeatureList.indexOf(rawToken);
                w.setRawToken(rawToken);
                w.setRawIndex(rawIndex);
                wordList.add(w);

            }
        }

        return spVct;
    }

    public void featureSelection(String location, String featureSelection, double startProb, double endProb, int maxDF, int minDF, String rawFeatureFile) throws FileNotFoundException {
        FeatureSelector selector = new FeatureSelector(startProb, endProb, maxDF, minDF);

        System.out.println("*******************************************************************");
        if (featureSelection.equals("DF"))
            selector.DF(m_featureStat);
        else if (featureSelection.equals("IG"))
            selector.IG(m_featureStat, m_classMemberNo);
        else if (featureSelection.equals("MI"))
            selector.MI(m_featureStat, m_classMemberNo);
        else if (featureSelection.equals("CHI"))
            selector.CHI(m_featureStat, m_classMemberNo);

        m_featureNames = selector.getSelectedFeatures();

        System.out.println(m_featureNames.size() + " features are selected!");

        // need some redesign of the current awkward procedure for feature selection and feature vector construction!!!!
        //clear memory for next step feature construction
//		reset();
//		LoadCV(location);//load the selected features
    }

    public void filterFeaturesbyGlove (String featureLocation, String featureSelection, double startProb, double endProb, int maxDF, int minDF, String rawFeatureFile, String gloveFile)throws FileNotFoundException{
        loadGloveVec(gloveFile); //filter features whose corresponding rawFeature do not have embeddings and filter rawFeatures whose features are not selected.

        System.out.println("m_featureName size\t"+m_featureNames.size());
        SaveCV(featureLocation, featureSelection, startProb, endProb, maxDF, minDF, rawFeatureFile);
    }

    protected boolean AnalyzeDocByStn(_Doc doc, String[] sentences) {
        TokenizeResult result;
        int y = doc.getYLabel(), index = 0;
        HashMap<Integer, Double> spVct = new HashMap<Integer, Double>(); // Collect the index and counts of features.
        ArrayList<_Stn> stnList = new ArrayList<_Stn>(); // sparse sentence feature vectors
        double stopwordCnt = 0, rawCnt = 0;

        ArrayList<_Word> wordList = new ArrayList<_Word>();

        for(String sentence : sentences) {
            result = TokenizerNormalizeStemmer(sentence);// Three-step analysis.
            String[] tokens = result.getTokens();
            String[] rawTokens = result.getRawTokens();

            ArrayList<_Word> sentence_wordList = new ArrayList<_Word>();
            HashMap<Integer, Double> sentence_vector = constructSpVct(tokens, rawTokens, y, null, sentence_wordList);// construct bag-of-word vector based on normalized tokens

//            HashMap<Integer, Double> sentence_vector = constructSpVct(rawTokens, rawTokens, y, null, wordList);// construct bag-of-word vector based on normalized tokens

            if (sentence_vector.size()>2) {//avoid empty sentence
                String[] posTags;
                if(m_tagger==null)
                    posTags = null;
                else
                    posTags = m_tagger.tag(result.getRawTokens());

                stnList.add(new _Stn(index, Utils.createSpVct(sentence_vector), result.getRawTokens(), posTags, sentence));
                Utils.mergeVectors(sentence_vector, spVct);

                for(_Word sentence_Word:sentence_wordList){
                    wordList.add(sentence_Word);
                }

                stopwordCnt += result.getStopwordCnt();
                rawCnt += result.getRawCnt();
            }
            index ++;
        } // End For loop for sentence

        //the document should be long enough
        if (spVct.size()>=m_lengthThreshold && stnList.size()>=m_stnSizeThreshold) {
            doc.createSpVct(spVct);
            doc.setStopwordProportion(stopwordCnt/rawCnt);
            doc.setSentences(stnList);
            doc.setWords(wordList);

            for(_Word w:doc.getWords()){
                String rawToken = w.getRawToken();
                if(!m_isCVLoaded) {
                    if (!m_rawTokenStrList.contains(rawToken)) {
                        m_rawTokenStrList.add(rawToken);
                    }
                }else{
                    if(!m_rawFeatureList.contains(rawToken)){
                        System.out.println("error raw feature\t"+rawToken);
                    }
                }
            }

            m_corpus.addDoc(doc);
            m_classMemberNo[y] ++;

            if (m_releaseContent)
                doc.clearSource();
            return true;
        } else {
            /****Roll back here!!******/
            rollBack(spVct, y);
            return false;
        }
    }

    protected void SaveCV(String featureLocation, String featureSelection, double startProb, double endProb, int maxDF, int minDF, String rawFeatureFile) throws FileNotFoundException {
        if (featureLocation==null || featureLocation.isEmpty())
            return;
        String feature;
        System.out.format("Saving controlled vocabulary to %s...\n", featureLocation);
        PrintWriter writer = new PrintWriter(new File(featureLocation));
        //print out the configurations as comments
        writer.format("#NGram:%d\n", m_Ngram);
        writer.format("#Selection:%s\n", featureSelection);
        writer.format("#Start:%f\n", startProb);
        writer.format("#End:%f\n", endProb);
        writer.format("#DF_MaxCut:%d\n", maxDF);
        writer.format("#DF_MinCut:%d\n", minDF);

        //print out the features
        int totalOutputFeature = 0;
        System.out.println("feature size\t"+m_featureNames.size());
        System.out.println("feature names============");
        for(int i=0; i<m_featureNames.size(); i++){
            feature = m_featureNames.get(i);
            System.out.println("feature names\t"+feature);
        }
        System.out.println("feature names============");

        System.out.println("output feature============");
        for(int i=0; i<m_outputFeatureList.size(); i++){
            feature = m_outputFeatureList.get(i);
            System.out.println("output feature\t"+feature);
            writer.println(feature);
        }
        System.out.println("output feature============");

        System.out.println("outputFeature Size\t"+m_outputFeatureList.size());

        writer.close();

        PrintWriter rawFeaturePW = new PrintWriter(new File(rawFeatureFile));

        for(String rawFeature:m_rawFeatureList){
            rawFeaturePW.println(rawFeature);
        }
        rawFeaturePW.flush();
        rawFeaturePW.close();

        System.out.println("m_rawFeatureList Size\t"+m_rawFeatureList.size());
    }

    public void loadGloveVec(String gloveFile){
        try{
            String tmpTxt;
            String[] lineContainer;
            double[] featureVecEle;
            int tid = 0;

            ArrayList<String> gloveRawTokenList = new ArrayList<String>();

            BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(gloveFile), "UTF-8"));

//            System.out.println("raw tokens from glove================");

            while((tmpTxt=br.readLine())!=null){
                tmpTxt = tmpTxt.trim();
                if(tmpTxt.isEmpty())
                    continue;

                lineContainer = tmpTxt.split(" ");
                String rawWordStr = Normalize(lineContainer[0]);

                if(!m_isCVLoaded) {
                    //filter word embeddings
                    if (!m_rawTokenStrList.contains(rawWordStr))
                        continue;

                    double[] tokenEmbeddingVec = new double[lineContainer.length - 1];
                    rawToken rawTokenObj = new rawToken(rawWordStr);

                    for (int i = 1; i < lineContainer.length; i++) {
                        tokenEmbeddingVec[i - 1] = Double.parseDouble(lineContainer[i]);
                    }

                    TokenizeResult resultToken = TokenizerNormalizeStemmer(rawWordStr);
                    if (resultToken.getTokens().length == 0)
                        continue;
                    String wordStr = resultToken.getTokens()[0];
                    rawTokenObj.setEmbeddingVec(tokenEmbeddingVec);

                    // in featureNames and have word embedding
                    if (m_featureNames.contains(wordStr)) {
                        if(!m_outputFeatureList.contains(wordStr))
                            m_outputFeatureList.add(wordStr);
                        if(!m_rawFeatureList.contains(rawWordStr))
                            m_rawFeatureList.add(rawWordStr);
                    }
                }else{
                    if (!m_rawFeatureList.contains(rawWordStr))
                        continue;

                    double[] tokenEmbeddingVec = new double[lineContainer.length - 1];
                    rawToken rawTokenObj = new rawToken(rawWordStr);

                    for (int i = 1; i < lineContainer.length; i++) {
                        tokenEmbeddingVec[i - 1] = Double.parseDouble(lineContainer[i]);
                    }

                    TokenizeResult resultToken = TokenizerNormalizeStemmer(rawWordStr);
                    if (resultToken.getTokens().length == 0){
                        System.out.println("error"+rawWordStr);
                        continue;
                    }

                    rawTokenObj.setEmbeddingVec(tokenEmbeddingVec);

                    m_rawTokenMap.put(rawWordStr, rawTokenObj);
                }
            }

//            System.out.println("================");

//            System.out.println("glove raw tokens========");
//            for(String rawToken:gloveRawTokenList){
//                System.out.println("glove tokens\t"+rawToken);
//            }
//            System.out.println("glove raw tokens========");
//
//            System.out.println("raw tokens =============");
//            for(String rawToken:m_rawTokenStrList){
//                System.out.println("raw tokens\t"+rawToken);
//            }
//            System.out.println("====================");

        }catch (Exception e){
            e.printStackTrace();
        }
    }

    public void simWords4Corpus(String filePrefix, String gloveFile){
        loadGloveVec(gloveFile);

        String wordSimFile = filePrefix+"rawFeatureSim_Tech.txt";
        ArrayList<Double> simList = new ArrayList<Double>();

        try{
            PrintWriter pw = new PrintWriter(new File(wordSimFile));
            System.out.println("rawTokenList len\t"+m_rawTokenStrList.size());
            for(int i=0; i<m_rawFeatureList.size(); i++){
                String rawFeature = m_rawFeatureList.get(i);
                if(!m_rawTokenMap.containsKey(rawFeature)){
                    System.out.println("sim word 4corpus error raw feature no word embedding\t"+rawFeature);
                    continue;
                }
                pw.print(rawFeature+"\t");
            }
            pw.println();

            System.out.println("m_rawTokenMap size\t"+m_rawTokenMap.size());

            for(int i=0; i<m_rawFeatureList.size(); i++){

                String wStrRow = m_rawFeatureList.get(i);
                if(!m_rawTokenMap.containsKey(wStrRow))
                    continue;
                rawToken rawTokenObjRow = m_rawTokenMap.get(wStrRow);
//                if(rawTokenObjRow.getEmbeddingVec()==null) {
//                    System.out.println(wStrRow + "\tno embedding vec");
//                    continue;
//                }

                for(int j=0; j<m_rawFeatureList.size(); j++){
                    String wStrCol = m_rawFeatureList.get(j);

                    if(!m_rawTokenMap.containsKey(wStrCol))
                        continue;
                    rawToken rawTokenObjCol = m_rawTokenMap.get(wStrCol);
//                    if(rawTokenObjCol.getEmbeddingVec()==null) {
//                        System.out.println(wStrCol + "\tno embedding vec");
//                        continue;
//                    }

                    double cosSim = Utils.cosine(rawTokenObjCol.getEmbeddingVec(),rawTokenObjRow.getEmbeddingVec());
                    //                wStatRow.m_wordSimMap.put(wStrCol, cosSim);
//                    pw.print(cosSim+"\t");
                    simList.add(cosSim);
                }
                pw.println();
            }

            pw.flush();
            pw.close();
        }catch(Exception e){
            e.printStackTrace();
        }


        Collections.sort(simList, Collections.reverseOrder());
        System.out.println("number of sim pairs\t"+simList.size());
        System.out.println("max sim\t"+Collections.max(simList)+"\t"+simList.get(0));
        System.out.println("min sim\t"+Collections.min(simList)+"\t"+simList.get(simList.size()-1));
        int mediumIndex = (int)simList.size()/2;
        int top10Index = (int)(simList.size()*0.1);
        int top20Index = (int)(simList.size()*0.2);
        int top40Index = (int)(simList.size()*0.4);
        System.out.println("medium \t"+simList.get(mediumIndex));
        System.out.println("top 10 sim \t"+simList.get(top10Index));
        System.out.println("top 20 sim \t"+simList.get(top20Index));
        System.out.println("top 40 sim \t"+simList.get(top40Index));

    }

    protected TokenizeResult TokenizerNormalizeStemmer(String source){
        String[] tokens = Tokenizer(source); //Original tokens.
        TokenizeResult result = new TokenizeResult(tokens);
        String[] rawTokens = result.getRawTokens();

        //Normalize them and stem them.
        for(int i = 0; i < tokens.length; i++)
            tokens[i] = SnowballStemming(Normalize(tokens[i]));

        LinkedList<String> Ngrams = new LinkedList<String>();
        int tokenLength = tokens.length, N = m_Ngram;

        ArrayList<String> rawTokenNGrams = new ArrayList<String>();

        for(int i=0; i<tokenLength; i++) {
            String token = tokens[i];
            String rawToken = rawTokens[i];
            boolean legit = isLegit(token);
            if (legit) {
                Ngrams.add(token);//unigram
                rawTokenNGrams.add(rawToken);
            }
            else
                result.incStopwords();

            //N to 2 grams
            if (!isBoundary(token)) {
                for(int j=i-1; j>=Math.max(0, i-N+1); j--) {
                    if (isBoundary(tokens[j]))
                        break;//touch the boundary

                    token = tokens[j] + "-" + token;
                    legit &= isLegit(tokens[j]);
                    if (legit)//at least one of them is legitimate
                        Ngrams.add(token);
                }
            }
        }

        result.setTokens(Ngrams.toArray(new String[Ngrams.size()]));
        result.setRawTokens(rawTokenNGrams.toArray(new String[rawTokenNGrams.size()]));
        return result;
    }

    public void loadWordSim4Corpus(String wordSimFileName, double[][]m_wordSimMatrix, double[] m_wordSimVec){
        double simThreshold = -1;
        if(wordSimFileName == null||wordSimFileName.isEmpty()){
            return;
        }

        try{
            double maxSim = -2;
            double minSim = 2;

            int rawFeatureSize = m_wordSimMatrix.length;
            for(int rawV=0; rawV<rawFeatureSize; rawV++) {
                Arrays.fill(m_wordSimMatrix[rawV], 0);
                Arrays.fill(m_wordSimVec, 0);
            }

            String tmpTxt;
            String[] lineContainer;

            HashMap<String, Integer> featureNameIndex = new HashMap<String, Integer>();
            for(int i=0; i<m_rawFeatureList.size(); i++){
                featureNameIndex.put(m_rawFeatureList.get(i), i);
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
//            normalizeSimByExp(simThreshold, rawFeatureSize, m_wordSimMatrix, m_wordSimVec);
        }catch (Exception e){
            e.printStackTrace();
        }
    }

    public void normalizeSimByExp(double threshold, int rawFeatureSize, double[][]m_wordSimMatrix, double[]m_wordSimVec){
        for(int i=0; i<rawFeatureSize; i++) {
            m_wordSimVec[i] = 0;
            for (int j = 0; j < rawFeatureSize; j++) {
                double normalizedSim = Math.exp(m_wordSimMatrix[i][j] );

                if(normalizedSim< threshold)
                    m_wordSimMatrix[i][j] = 0;
                else
                    m_wordSimMatrix[i][j] = normalizedSim;
                m_wordSimVec[i] += normalizedSim;
            }
        }
    }

    public void articleSim4Corpus(String filePrefix, String wordSimFileName){

        Random m_rand = new Random();

        int rawFeatureSize = m_rawFeatureList.size();
        double[][] m_wordSimMatrix = new double[rawFeatureSize][rawFeatureSize];
        double[] m_wordSimVec = new double[rawFeatureSize];

        loadWordSim4Corpus(wordSimFileName, m_wordSimMatrix, m_wordSimVec);

        ArrayList<_ParentDoc> pDocList = new ArrayList<_ParentDoc>();

        int pDocIndex = 0;
        for(_Doc d: m_corpus.getCollection()){
//            if(pDocIndex < 3) {
                if (d instanceof _ParentDoc) {
//                    pDocIndex += 1;
                    pDocList.add((_ParentDoc) d);
                }
//            }
        }


        try {
            File resultFolder = new File(filePrefix+"/articleSim/");

            if(!resultFolder.exists()){
                System.out.println("creating root directory"+resultFolder);
                resultFolder.mkdir();
            }

            for(_ParentDoc pDoc : pDocList) {
                String parentSimFile = pDoc.getName() + ".txt";
                PrintWriter parentOut = new PrintWriter(new File(
                        resultFolder, parentSimFile));

                ArrayList<Integer> featureList = new ArrayList<Integer>();
                for(_ChildDoc cDoc: pDoc.m_childDocs){
                    for(_Word cWord:cDoc.getWords()){
                        int cFeatureIndex= cWord.getRawIndex();
                        if(!featureList.contains(cFeatureIndex))
                            featureList.add(cFeatureIndex);
                    }
                }

                for(int rawFeatureIndex=0; rawFeatureIndex<featureList.size(); rawFeatureIndex++){
                    String rawFeature = m_rawFeatureList.get(rawFeatureIndex);
                    parentOut.print(rawFeature+"\t");

                    for(_Word pWord: pDoc.getWords()){
                        int pRawFeatureIndex = pWord.getRawIndex();
                        String pWordStr = pWord.getRawToken();
                        double cosineSim = m_wordSimMatrix[rawFeatureIndex][pRawFeatureIndex];
                        parentOut.print(pWordStr+":"+cosineSim+"\t");
                    }

                    parentOut.println();
                }

                parentOut.println("Here is the boundary");

                for(int rawFeatureIndex=0; rawFeatureIndex<m_rawFeatureList.size(); rawFeatureIndex++){
                    if(featureList.contains(rawFeatureIndex))
                        continue;
                    String rawFeature = m_rawFeatureList.get(rawFeatureIndex);
                    parentOut.print(rawFeature+"\t");

                    for(_Word pWord: pDoc.getWords()){
                        int pRawFeatureIndex = pWord.getRawIndex();
                        String pWordStr = pWord.getRawToken();
                        double cosineSim = m_wordSimMatrix[rawFeatureIndex][pRawFeatureIndex];
                        parentOut.print(pWordStr+":"+cosineSim+"\t");
                    }

                    parentOut.println();
                }

                parentOut.flush();
                parentOut.close();
            }
        }catch (Exception e){
            e.printStackTrace();
        }

    }

    public void randArticle(String filePrefix){

        Random m_rand = new Random();

        ArrayList<_ParentDoc> pDocList = new ArrayList<_ParentDoc>();

        for(_Doc d: m_corpus.getCollection()){
            if(d instanceof _ParentDoc){
                if(d.getName().equals("444")){
                    pDocList.add((_ParentDoc)d);
                }
            }
        }

        int randArticleIndex = m_rand.nextInt(pDocList.size());
        _ParentDoc pDoc = pDocList.get(randArticleIndex);

        try {
            double totalSim = 0;
            double avgSim = 0;

            String parentSimFile = pDoc.getName()+".txt";
            PrintWriter parentOut = new PrintWriter(new File(
                    filePrefix, parentSimFile));

            for (String rawToken:m_rawTokenMap.keySet()) {
                rawToken rawTokenObj = m_rawTokenMap.get(rawToken);
                parentOut.print(rawToken + "\t");

                for (_Word pWord : pDoc.getWords()) {
                    String pWordStr = pWord.getRawToken();
                    rawToken pWordTokenObj = m_rawTokenMap.get(pWordStr);
                    double cosineSim = Utils.cosine(pWordTokenObj.getEmbeddingVec(), rawTokenObj.getEmbeddingVec());
                    parentOut.print(pWordStr + ":" + cosineSim + "\t");
                }

                parentOut.println();
            }
            parentOut.flush();
            parentOut.close();

        }catch (Exception e){
            e.printStackTrace();
        }

    }

    public void randOutputSim4Comment(String filePrefix, String wordSimFileName){
//        int vocabulary_size = m_featureNames.size();

//        double[][] m_wordSimMatrix;
//        double[] m_wordSimVec;

//        m_wordSimMatrix = new double[vocabulary_size][vocabulary_size];
//        m_wordSimVec = new double[vocabulary_size];
//        loadWordSim4Corpus(wordSimFileName, m_wordSimMatrix, m_wordSimVec, vocabulary_size);

        Random m_rand = new Random();

        ArrayList<_ParentDoc> pDocList = new ArrayList<_ParentDoc>();

        for(_Doc d: m_corpus.getCollection()){
            if(d instanceof _ParentDoc)
                if(d.getName().equals("444"))
                    pDocList.add((_ParentDoc) d);
        }

        int randArticleIndex = m_rand.nextInt(pDocList.size());
        _ParentDoc pDoc = pDocList.get(randArticleIndex);

        try {
            for (_ChildDoc cDoc : pDoc.m_childDocs) {
                String commentSimFile = cDoc.getName()+".txt";
                PrintWriter childOut = new PrintWriter(new File(
                        filePrefix, commentSimFile));
                for(_Word cWord: cDoc.getWords()){
                    String cWordStr = cWord.getRawToken();
                    rawToken cWordTokenObj = m_rawTokenMap.get(cWordStr);
                    childOut.print(cWordStr+"\t");

                    for(_Word pWord:pDoc.getWords()){
                        String pWordStr = pWord.getRawToken();
                        rawToken pWordTokenObj = m_rawTokenMap.get(pWordStr);
                        double cosineSim = Utils.cosine(cWordTokenObj.getEmbeddingVec(), pWordTokenObj.getEmbeddingVec());
                        childOut.print(pWordStr+":"+cosineSim+"\t");
                    }
                    childOut.println();
                }

                childOut.flush();
                childOut.close();
            }
        }catch (Exception e){
            e.printStackTrace();
        }
    }

    public _Corpus getCorpus() {
        //store the feature names into corpus
        m_corpus.setFeatures(m_featureNames);
        m_corpus.setFeatureStat(m_featureStat);
        m_corpus.setMasks(); // After collecting all the documents, shuffle all the documents' labels.
        m_corpus.setContent(!m_releaseContent);
        m_corpus.setRawFeatures(m_rawFeatureList);
        return m_corpus;
    }

}
