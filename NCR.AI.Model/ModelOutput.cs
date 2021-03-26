using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.Data;

namespace NCR.AI.Model
{
    public class ModelOutput
    {
        [ColumnName("PredictedLabel")]
        public string Category { get; set; }

        public float[] Score { get; set; }
    }
}
