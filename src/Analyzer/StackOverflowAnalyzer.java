package Analyzer;

import java.io.*;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Locale;

public class StackOverflowAnalyzer {
    class _Question{
        int m_qid;
        int m_uid;
        long m_time;
        String m_title;
        String m_body;
        ArrayList<_Answer> m_answers = new ArrayList<>();

        public _Question(int qid, int uid){
            m_qid = qid;
            m_uid = uid;
        }

        public _Question(int qid, int uid, long time, String title, String body){
            m_qid = qid;
            m_uid = uid;
            m_time = time;
            m_title = title;
            m_body = body;
        }

        protected void addOneAnswer(_Answer a){
            m_answers.add(a);
        }

    }

    class _Answer{
        int m_aid;
        int m_qid;
        int m_uid;
        long m_time;
        String m_body;

        public _Answer(int aid, int qid, int uid){
            m_aid = aid;
            m_qid = qid;
            m_uid = uid;
        }

        public _Answer(int aid, int qid, int uid, long time, String str){
            m_aid = aid;
            m_qid = qid;
            m_uid = uid;
            m_time = time;
            m_body = str;
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
    HashMap<Integer, HashSet<Integer>> m_userConnectionMap = new HashMap<>();

    SimpleDateFormat m_format = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss'Z'", Locale.US);

    // init the stat map
    public StackOverflowAnalyzer(){
        for(int i=2; i<=10; i++){
            m_docSizeUserIdMap.put(i, new HashSet<>());
        }
    }

    protected void loadQuestions(String filename) {
        try {
            BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename)));
            int count = 0;
            // skip the first line as it is column name
            String line = reader.readLine();

            // Id,OwnerUserId,CreationDate,ClosedDate,Score
            while ((line = reader.readLine()) != null) {
                int bodyStart = line.indexOf("\"");
                if(bodyStart < 0) {
                    count++;
                    continue;
                }

                String[] strs = line.substring(0, bodyStart-1).split(",");
                if(strs.length != 5)
                    continue;

                // index,Id,OwnerUserId,CreationDate,Score,Title,Body
                int qid = Integer.valueOf(strs[0]);
                int uid = Integer.valueOf(strs[1]);

                //2008-08-01T13:57:07Z
                long time = m_format.parse(strs[2]).getTime()/1000;
                String title = strs[4];
                String body = line.substring(bodyStart+1, line.length()-1);
                _Question q = new _Question(qid, uid, time, title, body);
                m_questionMap.put(qid, q);
                if(!m_userMap.containsKey(uid))
                    m_userMap.put(uid, new _User(uid));
                m_userMap.get(uid).addOneQuestion(q);
            }
            reader.close();
            System.out.format("[Info]Finish loading the questions, %d questions are missing!\n", count);

        } catch(NumberFormatException e){
            e.printStackTrace();
        } catch(IOException e) {
            System.err.format("[Error]Failed to open file %s!!", filename);
            e.printStackTrace();
        } catch(ParseException e){
            System.err.format("[Error] Error in parse date!");
            e.printStackTrace();
        }
    }

    protected void loadAnswers(String filename){
        try {
            BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename)));

            // skip the first line as it is column name
            String line = reader.readLine();

            //index,Id,OwnerUserId,CreationDate,ParentId,Score,Body
            while ((line = reader.readLine()) != null) {
                int bodyStart = line.indexOf("\"");
                if(bodyStart < 0)
                    continue;
                String[] strs = line.substring(0, bodyStart-1).split(",");

                if(strs.length != 5) continue;
                int aid = Integer.valueOf(strs[0]);
                int uid = Integer.valueOf(strs[1]);
                long time = m_format.parse(strs[2]).getTime()/1000;
                int qid = Integer.valueOf(strs[3]);

                String body = line.substring(bodyStart+1, line.length()-1);
                // if the question does not exist, ignore the answer
                if(!m_questionMap.containsKey(qid))
                    continue;

                // else new a answer
                _Answer a = new _Answer(aid, qid, uid, time, body);
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
        }catch(ParseException e) {
            System.err.format("[Error] Error in parse date!");
            e.printStackTrace();
        }
    }

    protected void loadQuestionsWithoutText(String filename) {
        try {
            BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename)));
            // skip the first line as it is column name
            String line = reader.readLine();

            // Id,OwnerUserId,CreationDate,ClosedDate,Score
            while ((line = reader.readLine()) != null) {

                String[] strs = line.split(",");
                if(strs.length != 3)
                    continue;

                // index,Id,OwnerUserId,CreationDate,Score,Title,Body
                int qid = Integer.valueOf(strs[1]);
                int uid = Integer.valueOf(strs[2]);

                _Question q = new _Question(qid, uid);
                m_questionMap.put(qid, q);
                if(!m_userMap.containsKey(uid))
                    m_userMap.put(uid, new _User(uid));
                m_userMap.get(uid).addOneQuestion(q);
            }
            reader.close();
            System.out.format("[Info]Finish loading %d questions!\n", m_questionMap.size());
        } catch(IOException e) {
            System.err.format("[Error]Failed to open file %s!!", filename);
            e.printStackTrace();
        }
    }

    protected void loadAnswersWithoutText(String filename){
        try {
            BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename)));
            int answerNoQuestionCount = 0, totalAnswerCount = 0;
            // skip the first line as it is column name
            String line = reader.readLine();

            //index,Id,OwnerUserId,CreationDate,ParentId,Score,Body
            while ((line = reader.readLine()) != null) {
                totalAnswerCount++;
                String[] strs = line.split(",");

                if(strs.length != 4) continue;
                int aid = Integer.valueOf(strs[1]);
                int uid = Integer.valueOf(strs[2]);
                int qid = Integer.valueOf(strs[3]);

                // if the question does not exist, ignore the answer
                if(!m_questionMap.containsKey(qid)){
                    answerNoQuestionCount++;
                    continue;
                }

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
            System.out.format("[Info]Finish loading %d answers, (%d, %d) answers have/don't have corresponding questions!\n",
                    totalAnswerCount, m_answerMap.size(), answerNoQuestionCount);

        } catch(IOException e){
            System.err.format("[Error]Failed to open file %s!!", filename);
            e.printStackTrace();
        }
    }

    // select the users based on their doc size
    // construct the network based on the question and answer
    protected void calcNetworkStat(int threshold){

        System.out.format("\n======Current threshold is %d=======\n", threshold);
        m_userConnectionMap.clear();

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
        printUserIds("./data/stackOverflow_threshold_6.txt", candidates);
    }

    protected HashSet<Integer> filterUsers(int threshold){
        HashSet<Integer> uids = new HashSet<Integer>();
        for(int uid: m_userMap.keySet()){
            _User user = m_userMap.get(uid);
            if(user.getQuestionSize() + user.getAnswerSize() >= threshold){
                uids.add(uid);
            }
        }
        System.out.format("%d users have more than 10 docs!\n", uids.size());
        return uids;
    }
    // construct the network based on the selected users
    protected void constructNetwork(){
        m_userConnectionMap.clear();
        int answerNotInUserSetCount = 0;

        for(int uid: m_userMap.keySet()){
            _User user = m_userMap.get(uid);
            // build the table
            for(_Question q: user.m_questions){
                for(_Answer a: q.m_answers){
                    if(m_userMap.keySet().contains(a.m_uid)){
                        if(!m_userConnectionMap.containsKey(uid))
                            m_userConnectionMap.put(uid, new HashSet<>());
                        if(!m_userConnectionMap.containsKey(a.m_uid))
                            m_userConnectionMap.put(a.m_uid, new HashSet<>());
                        m_userConnectionMap.get(uid).add(a.m_uid);
                        m_userConnectionMap.get(a.m_uid).add(uid);
                    } else{
                        answerNotInUserSetCount++;
                    }
                }
            }
        }
        double avgConnectionSize = 0;
        int userNoConnectionCount = 0;

        for(int uid: m_userMap.keySet()){
            if(!m_userConnectionMap.containsKey(uid)){
                userNoConnectionCount++;
                continue;
            }
            avgConnectionSize += m_userConnectionMap.get(uid).size();
        }
        System.out.format("Total user size: %d, users with connections: %d.\n", m_userMap.size(),
                m_userConnectionMap.size());
        avgConnectionSize /= m_userMap.size();
        System.out.format("[Stat] %d users don't have any friends, avg friend size %.4f\n",
                userNoConnectionCount, avgConnectionSize);
        System.out.format("%d answers are not in the user set!\n", answerNotInUserSetCount);
    }


    // construct the network based on the selected users
    protected void constructNetworkWithFitleredUsers(HashSet<Integer> uids){
        m_userConnectionMap.clear();
        int answerNotInUserSetCount = 0;

        for(int uid: uids){
            _User user = m_userMap.get(uid);
            // build the table
            for(_Question q: user.m_questions){
                for(_Answer a: q.m_answers){
                    if(uids.contains(a.m_uid)){
                        if(!m_userConnectionMap.containsKey(uid))
                            m_userConnectionMap.put(uid, new HashSet<>());
                        if(!m_userConnectionMap.containsKey(a.m_uid))
                            m_userConnectionMap.put(a.m_uid, new HashSet<>());
                        m_userConnectionMap.get(uid).add(a.m_uid);
                        m_userConnectionMap.get(a.m_uid).add(uid);
                    } else{
                        answerNotInUserSetCount++;
                    }
                }
            }
        }
        double avgConnectionSize = 0;
        int userNoConnectionCount = 0;

        for(int uid: uids){
            if(!m_userConnectionMap.containsKey(uid)){
                userNoConnectionCount++;
                continue;
            }
            avgConnectionSize += m_userConnectionMap.get(uid).size();
        }
        System.out.format("Total user size: %d, users with connections: %d.\n", uids.size(),
                m_userConnectionMap.size());
        avgConnectionSize /= uids.size();
        System.out.format("[Stat] %d users don't have any friends, avg friend size %.4f\n",
                userNoConnectionCount, avgConnectionSize);
        System.out.format("%d answers are not in the user set!\n", answerNotInUserSetCount);
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
    protected void saveNetwork(String filename){
        try{
            PrintWriter writer = new PrintWriter(new File(filename));
            for(int uid: m_userConnectionMap.keySet()){
                writer.write(uid+"\t");
                for(int frd: m_userConnectionMap.get(uid))
                    writer.write(frd+"\t");
                writer.write("\n");

            }
            writer.close();
            System.out.format("Finish writing %d users friends!", m_userConnectionMap.size());
        } catch(IOException e){
            e.printStackTrace();
        }
    }

    protected void printUserIds(String filename, HashSet<Integer> userIds){
        try{
            PrintWriter writer = new PrintWriter(new File(filename));
            for(int uid: userIds){
                writer.write(uid+"\n");
            }
            writer.close();
        } catch(IOException e){
            e.printStackTrace();
        }
    }

    protected void calcNuQuestionsNotAnswered(){
        int count = 0;
        for(int qid: m_questionMap.keySet()){
            if(m_questionMap.get(qid).m_answers.size() == 0)
                count++;
        }
        System.out.format("%d/%d questions are not answered\n", count, m_questionMap.size());
    }

    protected HashSet<Integer> sample(HashSet<Integer> uids){
        HashSet<Integer> sampledUserIds = new HashSet<>();
        for(int uid: uids){
            if(Math.random() > 0.55){
                sampledUserIds.add(uid);
            }
        }
        return sampledUserIds;
    }

    public static void main(String[] args) {
        String questionFile = "/Users/lin/Documents/Lin'sWorkSpace/Notebook/data/Questions_Network.csv";
        String answerFile = "/Users/lin/Documents/Lin'sWorkSpace/Notebook/data/Answers_Network.csv";

        StackOverflowAnalyzer analyzer = new StackOverflowAnalyzer();
        analyzer.loadQuestionsWithoutText(questionFile);
        analyzer.loadAnswersWithoutText(answerFile);
        int threshold = 10;
        analyzer.filterUsers(threshold);

//        HashSet<Integer> sampledUserIds = analyzer.sample(analyzer.filterUsers(threshold));
//        analyzer.constructNetworkWithFitleredUsers(sampledUserIds);
//        analyzer.printUserIds("./data/StackOverflowUserIds_12k.txt", sampledUserIds);
//        analyzer.saveNetwork("./data/StackOverflowFriends_12k.txt");

    }
}
