import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

/* Normalize co-occurence matrix */
public class Normalize {

    public static class NormalizeMapper extends Mapper<LongWritable, Text, Text, Text> {

        @Override
        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {

            /*
              Input: movieA:movieB \t relation
              Output: relationship list for movieA
              Outputkey: movieA
              Outputvalue: movieB=relation
            */
            String[] movie_relation = value.toString().trim().split("\t");

            if (movie_relation.length < 2) {
                return;
            }

            String movieA = movie_relation[0].split(":")[0];
            String movieB = movie_relation[0].split(":")[1];
            String relation = movie_relation[1];
            context.write(new Text(movieA), new Text(movieB + "=" + relation));
        }
    }

    public static class NormalizeReducer extends Reducer<Text, Text, Text, Text> {
        @Override
        public void reduce(Text key, Iterable<Text> values, Context context)
                throws IOException, InterruptedException {

            /* Normalize each row of co-occurence matrix
             Inputkey: movieA
             Inputvalue: value=<movieB:relation, movieC:relation...>
             Outputkey: movieB
             Outputvalue: movieA = normalized relative relation
            */

            /* Compute denominator of each row */
            int denominator = 0;
            Map<String, Integer> map = new HashMap<String, Integer>();
            for (Text value:values) {
                /* value: movieB = relation */
                String[] movie_relation = value.toString().split("=");
                String movieB = movie_relation[0];
                int relation = Integer.parseInt(movie_relation[1]);
                map.put(movieB, relation);
                denominator += relation;
            }

            /* Normalization */
            for (Map.Entry<String, Integer> entry:map.entrySet()) {
                /* Outputkey: movieB
                   OutputValue: relation
                 */
                String outputKey = entry.getKey();
                double normalizedRelation = (double) entry.getValue() /denominator;
                String outputValue = key.toString() + "=" + normalizedRelation;
                context.write(new Text(outputKey), new Text(outputValue));
            }

            /* Output example: movieB \t movieA = 0.875 */
        }
    }

    public static void main(String[] args) throws Exception {

        Configuration conf = new Configuration();

        Job job = Job.getInstance(conf);
        job.setMapperClass(NormalizeMapper.class);
        job.setReducerClass(NormalizeReducer.class);

        job.setJarByClass(Normalize.class);

        job.setInputFormatClass(TextInputFormat.class);
        job.setOutputFormatClass(TextOutputFormat.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);

        TextInputFormat.setInputPaths(job, new Path(args[0]));
        TextOutputFormat.setOutputPath(job, new Path(args[1]));

        job.waitForCompletion(true);
    }
}
