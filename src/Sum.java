import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;

import java.io.IOException;

/* Add the element-wise multiplications in Multiplication.java by movieID */
public class Sum {

    public static class SumMapper extends Mapper<LongWritable, Text, Text, DoubleWritable> {

        @Override
        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            /*
            * Input: userID:movieID \t subSum
			  OutputKey: userID:movieID
			  OutputValue: subSum
            * */
            String[] key_value = value.toString().trim().split("\t");
            context.write(new Text(key_value[0]), new DoubleWritable(Double.parseDouble(key_value[1])));
        }
    }

    public static class SumReducer extends Reducer<Text, DoubleWritable, Text, DoubleWritable> {
        @Override
        public void reduce(Text key, Iterable<DoubleWritable> values, Context context)
                throws IOException, InterruptedException {

            /*
            Inputkey: userID:movieID
            InputValue: subSum iterable
            Output: sum over each unique usrID:movieID
           */
            double sum = 0;
            for (DoubleWritable value:values) {
                sum += value.get();
            }
            context.write(key, new DoubleWritable(sum));
        }
    }

    public static void main(String[] args) throws Exception {

        Configuration conf = new Configuration();

        Job job = Job.getInstance(conf);
        job.setMapperClass(SumMapper.class);
        job.setReducerClass(SumReducer.class);

        job.setJarByClass(Sum.class);

        job.setInputFormatClass(TextInputFormat.class);
        job.setOutputFormatClass(TextOutputFormat.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(DoubleWritable.class);

        TextInputFormat.setInputPaths(job, new Path(args[0]));
        TextOutputFormat.setOutputPath(job, new Path(args[1]));

        job.waitForCompletion(true);
    }
}
