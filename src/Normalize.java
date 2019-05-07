import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
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

public class Normalize {

    public static class NormalizeMapper extends Mapper<LongWritable, Text, Text, Text> {

        // map method
        @Override
        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {

            /*Input: movieA:movieB \t relation
              Output: relationship list for movieA
              Outputkey: movieA
              Outputvalue: movieB = relation
            */
            String[] movie_relation = value.toString().trim().split("\t");
            String[] movieA = movie_relation[0].split(":")[0];
            String[] movieB = movie_relation[0].split(":")[1];
            String[] relation = movie_relation[1];
            context.write(new Text(movieA), new Text(movieB + "=" + relation));
        }
    }

    public static class NormalizeReducer extends Reducer<Text, Text, Text, Text> {
        // reduce method
        @Override
        public void reduce(Text key, Iterable<Text> values, Context context)
                throws IOException, InterruptedException {

            /* Normalize each row of co-occurence matrix
             Inputkey: movieA
             Inputvalue: value=<movieB:relation, movieC:relation...>
             Outputkey: movieB
             Outputvalue: movieA = normalized relative relation
            */
            int denominator = 0;
            Map<String, Integer> map = new HashMap<String, Integer>();
            for (Text value:values) {
                /*value: movieB = relation */
                String movieB = value.split("=")[0];
                int relation = Integer.parseInt(value.toString().split("=")[1]);
                map.put(movieB, relation);
                denominator += relation;
            }

            for (Map.Entry<String, Integer> entry:map.entrySet()) {
                // entry key: movieB
                // entryValue: relation
                String outputKey = entry.getKey();
                double normalizeRelation = (double) entry.getValue() /denominator;
                String outputValue = key.toString() + "=" + normalizeRelation;
                context.write(new Text(outputKey), new Text(outputValue));
            }

            /* movieB \t movieA = 0.875 */
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
