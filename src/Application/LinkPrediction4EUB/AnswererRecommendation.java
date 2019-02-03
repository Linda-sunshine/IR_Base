package Application.LinkPrediction4EUB;

import utils.Utils;

import java.io.*;
import java.util.HashMap;

/**
 * @author Lin Gong (lg5bt@virginia.edu)
 * In this task, we would input user embedding and a set of questions' topic distributions.
 * We want to find out if the learned embedding can help us find the right person to answer the question.
 */
public class AnswererRecommendation extends LinkPredictionWithUserEmbedding {

    double m_alpha = 0.1;

    class _Question{
        String m_postId;
        String m_uId;
        double[] m_theta;

        public _Question(String qId, String uId){
            m_postId = qId;
            m_uId = uId;
        }

        public _Question(String qId, String uId, double[] theta){
            m_postId = qId;
            m_uId = uId;
            m_theta = theta;
        }

        public String getUserId(){
            return m_uId;
        }

        public void setTheta(double[] theta){
            m_theta = theta;
        }
        public double[] getTheta(){
            return m_theta;
        }
    }

    // the topic distribution for selected questions
    String m_model;
    double[][] m_Phi;
    protected HashMap<String, _Question> m_questionMap;
    protected HashMap<Integer, double[]> m_userEmbeddings;

    public AnswererRecommendation(String model){
        m_questionMap = new HashMap<>();
        m_userEmbeddings = new HashMap<>();
        m_model = model;
    }

    public void setAlpha(double v){
        m_alpha = v;
    }
    public void loadQuestionIds(String filename){
        try {
            File file = new File(filename);
            BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(file), "UTF-8"));
            String line;

            while ((line = reader.readLine()) != null) {
                // the first is uid, the second is qid
                String[] strs = line.trim().split("\\s+");
                String uId = strs[0], qId = strs[1];
                if(!m_idIndexMap.containsKey(uId)){
                    System.out.println("The user does not exist in the user set!");
                }
                // put the question in the map first
                m_questionMap.put(qId, new _Question(qId, uId));
            }
            reader.close();
            System.out.format("Finish loading %d question ids from %s.\n", m_questionMap.size(), filename);

        }catch(IOException e){
            e.printStackTrace();
        }
    }
    // load thetas from the file
    public void loadQuestionTopicDistributions(String filename){
        try {
            File file = new File(filename);
            BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(file), "UTF-8"));
            String line = reader.readLine();
            // the first line is question size * dim
            int size = Integer.valueOf(line.split("\\s+")[0]);

            while ((line = reader.readLine()) != null) {
                // the first is uid, the second is qid
                String[] strs = line.trim().split("\\s+");
                String qId = strs[0];
                if(!m_questionMap.containsKey(qId)){
                    System.out.println("The question does not exist in the question map!!");
                    continue;
                }
                double[] theta = new double[strs.length-1];
                for(int i=1; i<strs.length; i++) {
                    theta[i-1] = Double.valueOf(strs[i]);
                }
                m_questionMap.get(qId).setTheta(theta);
            }
            reader.close();
            System.out.format("Finish loading %d documents' thetas from %s.\n", m_questionMap.size(), filename);

        }catch(IOException e){
            e.printStackTrace();
        }
    }

    // load the topic embeddings from the file
    public void loadPhi(String filename){
        try{
            File file = new File(filename);
            BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(file), "UTF-8"));
            String line = reader.readLine();
            String[] strs = line.split("\\s+");
            int numTopics = Integer.valueOf(strs[0]);
            int dim = Integer.valueOf(strs[1]);
            m_Phi = new double[numTopics][dim];
            int index = 0;
            while ((line = reader.readLine()) != null) {
                strs = line.split("\\s+");
                double[] oneTopicEmbedding = new double[dim];
                for(int i=0; i<strs.length; i++){
                    oneTopicEmbedding[i] = Double.valueOf(strs[i]);
                }
                m_Phi[index++] = oneTopicEmbedding;
            }
            reader.close();
            System.out.format("Finish loading %d topic embeddings!\n", numTopics);
        } catch(IOException e){
            e.printStackTrace();
        }
    }

    @Override
    // load edges (interactions + non-interactions) for all the users
    public void loadTestOneEdges(String filename){
        int labelOne = 1, count = 0;
        try {
            m_testMap.clear();
            m_testIds.clear();
            File file = new File(filename);
            BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(file), "UTF-8"));
            String line;
            while ((line = reader.readLine()) != null) {
                String[] strs = line.trim().split("\\s+");
                String uId = strs[0], qId = strs[1];

                if(!m_questionMap.containsKey(qId))
                    System.out.println("The question does not exist !!!");
                _Question q = m_questionMap.get(qId);
                _Object4Link qi = new _Object4Link(qId);
                m_testMap.put(qId, qi);
                m_testIds.add(qId);
                // record testing interactions of ui or qi
                for(int j=1; j<strs.length; j++){
                    String ujId = strs[j];
                    int ujIdx = m_idIndexMap.get(ujId);
                    // the question belongs to user_i.
                    int uiIdx = m_idIndexMap.get(q.getUserId());
                    double sim = getSimilarity(qId, uiIdx, ujIdx);
                    qi.addOneEdge(new _Edge4Link(ujIdx, sim, labelOne));

                }
                count++;
            }
            reader.close();
            System.out.format("[Info]%d/%d users' interactions are loaded!!\n", m_testMap.size(), count);
        } catch(IOException e){
            e.printStackTrace();
        }
    }

    @Override
    // load non-interactions from file for all the users
    public void loadTestZeroEdges(String filename){
        int labelZero = 0, count = 0;
        try {
            File file = new File(filename);
            BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(file), "UTF-8"));
            String line;
            while ((line = reader.readLine()) != null) {
                String[] strs = line.trim().split("\\s+");
                String qId = strs[0];
                if(!m_testMap.containsKey(qId))
                    continue;
                _Question q = m_questionMap.get(qId);
                _Object4Link qi = m_testMap.get(qId);
                for(int j=1; j<strs.length; j++){
                    String ujId = strs[j];
                    int uiIdx = m_idIndexMap.get(q.getUserId());
                    int ujIdx = m_idIndexMap.get(ujId);
                    double sim = getSimilarity(qId, uiIdx, ujIdx);
                    qi.addOneEdge(new _Edge4Link(ujIdx, sim, labelZero));
                }
                count++;
            }
            reader.close();
            System.out.format("[Info]%d/%d users' non-interactions are loaded!!\n", m_testMap.size(), count);
        } catch(IOException e){
            e.printStackTrace();
        }
    }

    // for baseline models, the sim = theta * uj + ui * uj
    // for eub model, the sim = uj \Phi theta + uj * ui
    public double getSimilarity(String qId, int uiIdx, int ujIdx){
        double[] theta = m_questionMap.get(qId).getTheta();
        double sim;
        if(m_model.startsWith("EUB")){
            // we need to incorporate |Phi in order to compute it.
            double[] uPhi = new double[theta.length];
            for(int i=0; i<theta.length; i++){
                uPhi[i] = Utils.dotProduct(m_Phi[i], m_embeddings[ujIdx]);
            }
            sim = m_alpha * Utils.dotProduct(theta, uPhi) + (1- m_alpha) * super.getSimilarity(uiIdx, ujIdx);
//            System.out.println(sim);
        } else{
            sim = m_alpha * Utils.cosine(theta, m_embeddings[ujIdx]) + (1-m_alpha) * super.getSimilarity(uiIdx, ujIdx);
        }
        return sim;
    }

    // init recommendation
    public void initRecommendation(String idFile, String qIdFile, String embedFile, String questionFile, String testInterFile, String testNonInterFile) {

        super.loadUserIds(idFile);
        super.loadUserEmbedding(embedFile);
        loadQuestionIds(qIdFile);
        loadQuestionTopicDistributions(questionFile);

        calcSimilarity();
        // if no non-interactions are specified, we will consider all the other users as non-interactions
        // thus, non-interactions are know after loading interactions
        if(testNonInterFile == null){
            System.out.println("Non interation file is null!!");
            loadTestOneZeroEdges(testInterFile);
        } else{
            // otherwise, load one-edges ans zero-edges individually
            loadTestOneEdges(testInterFile);
            loadTestZeroEdges(testNonInterFile);
        }
    }

    //In the main function, we want to input the data and do adaptation
    public static void main(String[] args) {
        // the application is only performed on stackoverflow dataset
        String data = "YelpNew";
        int dim = 10;
//        String model = "CTR";
        String prefix = String.format("./data/CoLinAdapt/%s/AnswerRecommendation", data);

        String[] models = new String[]{"EUB_t10", "HFT", "LDA_Variational", "CTR"};

        double[] alphas = new double[11];
        for(int i=0; i<=10; i++){
            alphas[i] = i * 0.1;
        }
        double[][][] perfs = new double[alphas.length][models.length][2];
        for(int a=0; a<alphas.length; a++) {
//            System.out.format("-----------------current alpha=%.1f-------------------\n", alpha);

            for (int i = 0; i < models.length; i++) {
                String model = models[i];

                System.out.format("-----------------model %s-dim %d-------------------\n", model, dim);
                String idFile = String.format("/home/lin/DataWWW2019/UserEmbedding/%s_userids.txt", data);
                String questionIdFile = String.format("%s/StackOverflowSelectedQuestions.txt", prefix);

                String questionFile = String.format("%s/models/%s_theta_dim_%d.txt", prefix, model, dim);
                String embedFile = String.format("%s/models/%s_embedding_dim_%d.txt", prefix, model, dim);
                String interFile = String.format("%s/%sInteractions4Recommendations_test.txt", prefix, data);
                String nonInterFile = String.format("%s/%sNonInteractions_time_10_Recommendations.txt", prefix, data);
                String phiFile = String.format("%s/models/EUB_t10_Phi_dim_%d.txt", prefix, dim);
                AnswererRecommendation rec = new AnswererRecommendation(model);
                if (model.equals("EUB"))
                    rec.loadPhi(phiFile);
                rec.setAlpha(alphas[a]);
                rec.initRecommendation(idFile, questionIdFile, embedFile, questionFile, interFile, nonInterFile);
                rec.calculateAllNDCGMAP();
                perfs[a][i] = rec.calculateAvgNDCGMAP();

            }
        }

        for(int i=0; i<alphas.length; i++){
            System.out.format("\talpha=%.1f\t", alphas[i]);
        }
        System.out.println();
        for (int i = 0; i < models.length; i++) {
            System.out.print(models[i]+"\t");
            for(int j=0; j<alphas.length; j++){
                System.out.format("%.4f\t%.4f\t", perfs[j][i][0], perfs[j][i][1]);
            }
            System.out.println();
        }
    }
}
