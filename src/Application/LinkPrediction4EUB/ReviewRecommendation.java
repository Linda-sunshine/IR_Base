package Application.LinkPrediction4EUB;

import java.io.*;
import java.util.HashMap;

/**
 * @author Lin Gong (lg5bt@virginia.edu)
 */
public class ReviewRecommendation extends AnswererRecommendation {

    HashMap<String, String> m_uidQidxMap = new HashMap<>();
    public ReviewRecommendation(String model){
        super(model);
    }

    @Override
    public void loadQuestionIds(String filename){
        try {
            File file = new File(filename);
            BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(file), "UTF-8"));
            String line;
//            // the first line is question size * dim
//            int size = Integer.valueOf(line.split("\\s+")[0]);

            while ((line = reader.readLine()) != null) {
                // the first is uid, the second is qid
                String[] strs = line.trim().split("\\s+");
                String uId = strs[0], qIdx = strs[1];
                m_uidQidxMap.put(uId, qIdx);
                String qId = String.format("%s#%s", uId, qIdx);
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

    @Override
    // load edges (interactions + non-interactions) for all the users
    public void loadTestOneEdges(String filename){
        int labelOne = 1, count = 0, repeat = 0;
        try {
            m_testMap.clear();
            m_testIds.clear();
            File file = new File(filename);
            BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(file), "UTF-8"));
            String line;
            while ((line = reader.readLine()) != null) {
                String[] strs = line.trim().split("\\s+");
                String uId = strs[0];
                String qIdx = m_uidQidxMap.get(uId);
                String qId = String.format("%s#%s", uId, qIdx);
                if(!m_questionMap.containsKey(qId))
                    System.out.println("The question does not exist !!!");
                _Question q = m_questionMap.get(qId);
                String[] ss = qId.split("#");
                if(!q.getUserId().equals(ss[0]))
                    System.out.println("The owner of the question does not align with the user!!");
                _Object4Link qi = new _Object4Link(qId);
                m_testMap.put(qId, qi);
                m_testIds.add(qId);
                // record testing interactions of ui or qi
                for(int j=1; j<strs.length; j++){

                    String ujId = strs[j];
                    int ujIdx = m_idIndexMap.get(ujId);
                    // the question belongs to user_i.
                    int uiIdx = m_idIndexMap.get(q.getUserId());
                    if(uiIdx == ujIdx){
                        repeat++;
                        continue;
                    }
                    double sim = getSimilarity(qId, uiIdx, ujIdx);
                    qi.addOneEdge(new _Edge4Link(ujIdx, sim, labelOne));

                }
                count++;
            }
            reader.close();
            System.out.format("%d users have repeat user id.\n", repeat);
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
                String uId = strs[0];
                String qIdx = m_uidQidxMap.get(uId);
                String qId = String.format("%s#%s", uId, qIdx);
                if(!m_testMap.containsKey(qId))
                    continue;
                _Question q = m_questionMap.get(qId);
                _Object4Link qi = m_testMap.get(qId);
                for(int j=1; j<strs.length; j++){
                    String ujId = strs[j];
                    int uiIdx = m_idIndexMap.get(q.getUserId());
                    int ujIdx = m_idIndexMap.get(ujId);
                    if(uiIdx == ujIdx){
                        System.out.println("same user in the non-interaction!");
                        continue;
                    }
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

    //In the main function, we want to input the data and do adaptation
    public static void main(String[] args) {
        String data = "YelpNew";
        int dim = 10;
        String prefix = String.format("./data/CoLinAdapt/%s/AnswerRecommendation", data);

        String[] models = new String[]{"CTR"};// "HFT", "LDA_Variational", "CTR"};

        double[] alphas = new double[1];
        alphas[0] = 0.5;

//        for(int i=0; i<=10; i++){
//            alphas[i] = i * 0.1;
//        }
        double[][][] perfs = new double[alphas.length][models.length][2];
        for(int a=0; a<alphas.length; a++) {

            System.out.format("-----------------current alpha=%.1f-------------------\n", alphas[a]);

            for (int i = 0; i < models.length; i++) {
                String model = models[i];

                System.out.format("-----------------model %s-dim %d-------------------\n", model, dim);
                String idFile = String.format("/home/lin/DataWWW2019/UserEmbedding/%s_userids.txt", data);
                String questionIdFile = String.format("%s/%sSelectedQuestions.txt", prefix, data);

                String questionFile = String.format("%s/models/%s_theta_dim_%d.txt", prefix, model, dim);
                String embedFile = String.format("%s/models/%s_embedding_dim_%d.txt", prefix, model, dim);
                String interFile = String.format("%s/%sInteractions4Recommendations_test.txt", prefix, data);
                String nonInterFile = String.format("%s/%sNonInteractions_time_10_Recommendations.txt", prefix, data);
                String phiFile = String.format("%s/models/%s_Phi_dim_%d.txt", prefix, model, dim);
                ReviewRecommendation rec = new ReviewRecommendation(model);
                if (model.startsWith("EUB"))
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
