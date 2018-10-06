package Analyzer;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;

public class StackOverflowAnalyzer {
    class _Question{
        int m_qid;
        int m_uid;
        ArrayList<_Answer> m_answers = new ArrayList<>();

        public _Question(int qid, int uid){
            m_qid = qid;
            m_uid = uid;
        }

        protected void addOneAnswer(_Answer a){
            m_answers.add(a);
        }

    }

    class _Answer{
        int m_aid;
        int m_qid;
        int m_uid;

        public _Answer(int aid, int qid, int uid){
            m_aid = aid;
            m_qid = qid;
            m_uid = uid;
        }
    }

    class _User {

        int m_uid;
        ArrayList<_Question> m_questions = new ArrayList<>();
        ArrayList<_Answer> m_answers = new ArrayList<>();

        public _User(int uid){
            m_uid = uid;
        }

        protected void addOneQuestion(_Question q){
            m_questions.add(q);
        }

        protected void addOneAnswer(_Answer a){
            m_answers.add(a);
        }

        protected int getQuestionSize(){
            return m_questions.size();
        }

        protected int getAnswerSize(){
            return m_answers.size();
        }
    }

    HashMap<Integer, _Question> m_questionMap = new HashMap<>();
    HashMap<Integer, _User> m_userMap = new HashMap<>();
    HashMap<Integer, _Answer> m_answerMap = new HashMap<>();

    HashSet<Integer> m_userIdsBoth = new HashSet<Integer>();
    HashMap<Integer, HashSet<Integer>> m_docSizeUserIdMap = new HashMap<>();

    // init the stat map
    public StackOverflowAnalyzer(){
        for(int i=2; i<=10; i++){
            m_docSizeUserIdMap.put(i, new HashSet<>());
        }
    }
    protected void loadQuestions(String filename) {
        try {
            BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename)));

            // skip the first line as it is column name
            String line = reader.readLine();

            // Id,OwnerUserId,CreationDate,ClosedDate,Score
            while ((line = reader.readLine()) != null) {
                String[] strs = line.split(",");
                if(strs.length != 5) continue;

                int qid = Integer.valueOf(strs[1]);
                double rawUId = Double.valueOf(strs[2]);
                int uid = (int) rawUId;
                _Question q = new _Question(qid, uid);
                m_questionMap.put(qid, q);
                if(!m_userMap.containsKey(uid))
                    m_userMap.put(uid, new _User(uid));
                m_userMap.get(uid).addOneQuestion(q);
            }
            reader.close();
            System.out.println("[Info]Finish loading the questions!");

        } catch(IOException e) {
            System.err.format("[Error]Failed to open file %s!!", filename);
            e.printStackTrace();
        }
    }

    protected void loadAnswers(String filename){
        try {
            BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename)));

            // skip the first line as it is column name
            String line = reader.readLine();

            //Id,OwnerUserId,CreationDate,ParentId,Score
            while ((line = reader.readLine()) != null) {
                String[] strs = line.split(",");
                if(strs.length != 6) continue;
                int aid = Integer.valueOf(strs[1]);
                double rawUId = Double.valueOf(strs[2]);
                int uid = (int) rawUId;
                int qid = Integer.valueOf(strs[4]);

                // if the question does not exist, ignore the answer
                if(!m_questionMap.containsKey(qid))
                    continue;

                // else new a answer
                _Answer a = new _Answer(aid, qid, uid);
                m_answerMap.put(aid, a);

                // add the answer to the question
                m_questionMap.get(qid).addOneAnswer(a);

                // add the answer to the user
                if(!m_userMap.containsKey(uid))
                    m_userMap.put(uid, new _User(uid));
                m_userMap.get(uid).addOneAnswer(a);
            }
            reader.close();
            System.out.println("[Info]Finish loading the answers!");
        } catch(IOException e){
            System.err.format("[Error]Failed to open file %s!!", filename);
            e.printStackTrace();
        }
    }

    HashMap<Integer, HashSet<Integer>> m_userConnectionMap = new HashMap<>();
    HashMap<Integer, HashSet<Integer>> m_networkMap = new HashMap<>();

    // select the users based on their doc size
    // construct the network based on the question and answer
    protected void constructNetwork(int threshold){

        System.out.format("\n======Current threshold is %d=======\n", threshold);
        m_userConnectionMap.clear();
        m_networkMap.clear();

        // threshold = 2, filter users with 2 docs.
        // threshold = 3, filter users with 3 docs and 2 docs.
        HashSet<Integer> filterUsers = new HashSet<>();
        HashSet<Integer> filterUsersNoAnswers = new HashSet<>();

        if(threshold >= 2){
            for(int i= threshold; i>=2; i--){
                filterUsers.addAll(m_docSizeUserIdMap.get(i));
            }
        }

        // key: user_id, value: users who answered the current user's question
        int  userNoConnection = 0;
        double avgOneDirectionCount = 0;
        HashSet<Integer> candidates = new HashSet<>();
        for(int uid: m_userIdsBoth){
            if(!filterUsers.contains(uid))
                candidates.add(uid);
        }

        for(int uid: candidates){
            _User user = m_userMap.get(uid);
            m_userConnectionMap.put(uid, new HashSet<>());
            // build the table
            int oneDirectionCount = 0;
            for(_Question q: user.m_questions){
                for(_Answer a: q.m_answers){
                    if(candidates.contains(a.m_uid)){
                        oneDirectionCount++;
                        m_userConnectionMap.get(uid).add(a.m_uid);
                    }
                }
            }
            avgOneDirectionCount += oneDirectionCount;
            if(oneDirectionCount == 0){
                userNoConnection++;
                filterUsersNoAnswers.add(uid);
            }
        }

        avgOneDirectionCount /= candidates.size();
        System.out.format("%d/%d users don't have any questions answered, each user has %.2f avg connections\n",
                userNoConnection, candidates.size(), avgOneDirectionCount);


        for(int uid: filterUsersNoAnswers){
            candidates.remove(uid);
        }

        System.out.format("%d users are left.\n", candidates.size());

        avgOneDirectionCount = 0;
        userNoConnection = 0;
        m_userConnectionMap.clear();
        for(int uid: candidates){
            _User user = m_userMap.get(uid);
            m_userConnectionMap.put(uid, new HashSet<>());
            // build the table
            int oneDirectionCount = 0;
            for(_Question q: user.m_questions){
                for(_Answer a: q.m_answers){
                    if(candidates.contains(a.m_uid)){
                        oneDirectionCount++;
                        m_userConnectionMap.get(uid).add(a.m_uid);
                    }
                }
            }
            avgOneDirectionCount += oneDirectionCount;
            if(oneDirectionCount == 0){
                userNoConnection++;
            }
        }

        avgOneDirectionCount /= candidates.size();
        System.out.format("[After]%d/%d users don't have any questions answered, each user has %.2f avg connections\n",
                userNoConnection, candidates.size(), avgOneDirectionCount);

        for(int uid: m_userConnectionMap.keySet()){

            for(int ujd: m_userConnectionMap.get(uid)){
                if(m_userConnectionMap.get(ujd).contains(uid)){
                    if(!m_networkMap.containsKey(uid)){
                        m_networkMap.put(uid, new HashSet<>());
                    }
                    if(!m_networkMap.containsKey(ujd)){
                        m_networkMap.put(ujd, new HashSet<>());
                    }
                    m_networkMap.get(uid).add(ujd);
                    m_networkMap.get(ujd).add(uid);
                }
            }
        }

        double connectionAvg = 0;
        for(int uid: m_networkMap.keySet()){
            connectionAvg += m_networkMap.get(uid).size();
        }
        connectionAvg /= m_networkMap.size();
        System.out.format("%d users have friends, avg friend size is %.4f.\n", m_networkMap.size(), connectionAvg);
    }

    // calcualte the basic statistics of the user information
    protected void calculateStat(){

        double both = 0, both_q = 0, both_a = 0, both_qa = 0;
        int[] docSize = new int[11];
        int max_gloal = 0, max_local = 0;
        for(int uid: m_userMap.keySet()){
            _User cur = m_userMap.get(uid);
            int sum = cur.getAnswerSize() + cur.getQuestionSize();
            max_gloal = Math.max(max_gloal, sum);

            if(cur.getQuestionSize() != 0 && cur.getAnswerSize() != 0){
                m_userIdsBoth.add(uid);
                max_local = Math.max(max_local, sum);
                both++;
                both_q += cur.getQuestionSize();
                both_a += cur.getAnswerSize();
                both_qa += sum;
                if(sum <= 10) {
                    docSize[sum]++;
                    m_docSizeUserIdMap.get(sum).add(uid);
                }
            }

        }
        both_q /= both;
        both_a /= both;
        both_qa /= both;
        System.out.println("User doc size:");
        for(int d: docSize)
            System.out.format("%d\t", d);
        System.out.println();
        System.out.format("For users with q+a, question avg: %.3f, answer avg: %.3f, q+a avg: %.3f\n", both_q, both_a, both_qa);
    }

    protected void calcNuQuestionsNotAnswered(){
        int count = 0;
        for(int qid: m_questionMap.keySet()){
            if(m_questionMap.get(qid).m_answers.size() == 0)
                count++;
        }
        System.out.format("%d/%d questions are not answered\n", count, m_questionMap.size());
    }

    public static void main(String[] args) {
        String questionFile = "/Users/lin/Documents/Lin'sWorkSpace/Notebook/Questions_filter.csv";
        String answerFile = "/Users/lin/Documents/Lin'sWorkSpace/Notebook/Answers_filter.csv";

        StackOverflowAnalyzer analyzer = new StackOverflowAnalyzer();
        analyzer.loadQuestions(questionFile);
        analyzer.loadAnswers(answerFile);

        System.out.format("Total number of questions: %d, answers: %d, users: %d\n", analyzer.m_questionMap.size(),
                analyzer.m_answerMap.size(), analyzer.m_userMap.size());

//        analyzer.calcNuQuestionsNotAnswered();
        analyzer.calculateStat();

        analyzer.constructNetwork(1);
        analyzer.constructNetwork(2);
        analyzer.constructNetwork(3);
        analyzer.constructNetwork(4);
        analyzer.constructNetwork(5);
        analyzer.constructNetwork(6);
        analyzer.constructNetwork(7);
        analyzer.constructNetwork(8);
        analyzer.constructNetwork(9);
        analyzer.constructNetwork(10);



    }
}
