"""
Single destination constrained gravity model
"""
class SingleDest:
    self.NumModes=3 #const???
    
###############################################################################

    def __init__(self, i_var):
        self.TObs=[] #Data input to model
        self.cost=[] #cost matrix for zones in TObs

        self.isUsingConstraints = false
        self.constraints = [] #1 or 0 to indicate constraints for zones matching TObs - this applies to all modes

        self.TPred=[] #this is the output
        self.B=[] #this is the constraints output vector - this applies to all modes
        self.Beta=[] #Beta values for three modes - this is also output


"""
    calculateCBar
    Mean trips calculation
    @param name="Tij" NDArray
    @param name="cost" NDArray
    @returns float
"""
    def calculateCBar(self,Tij,cost):
        (M, N) = np.shape(Tij)
        CNumerator = 0.0
        CDenominator = 0.0
        for i in range(0,N):
            for j in range(0,N):
                CNumerator += Tij[i, j] * cost[i, j]
                CDenominator += Tij[i, j]
        float CBar = CNumerator / CDenominator

        return CBar

###############################################################################
"""
    run
    Not sure what this was - early run function?
    @returns nothing
"""
    def run():
        (M, N) = np.shape(TObs)
        
        #set up Beta for modes 0, 1 and 2 to 1.0f
        Beta = [1.0 for k in range(0,self.NumModes)]

        #work out Dobs and Tobs from rows and columns of TObs matrix
        #These don't ever change so they need to be outside the convergence loop
        DjObs = np.arange(N)
        OiObs = np.arrange(N)
        Sum=0.0

        #OiObs
        for i in range(0,N):
            Sum = 0.0
            for j in range(0,N):
                for k in range(0, self.NumModes):
                    Sum += TObs[k]._M[i, j];
            OiObs[i] = Sum

        #DjObs
        for j in range(0,N):
            Sum = 0.0
            for i in range(0,N):
                for k in range(0,self.NumModes):
                    Sum += TObs[k]._M[i, j];
            DjObs[j] = Sum

        #constraints initialisation
        B = new float[N];
        for (int i = 0; i < N; i++) B[i] = 1.0f; //hack
        float[] Z = new float[N];
        for (int j = 0; j < N; j++)
        {
            Z[j] = float.MaxValue;
            if (IsUsingConstraints)
            {
                if (Constraints[j] >= 1.0f) //constraints array taking place of Gj (Green Belt) in documentation
                {
                    #Gj=1 means 0.8 of MSOA land is green belt, so can't be built on
                    #set constraint to original Dj
                    Z[j] = DjObs[j];
                }
            }
        }
        #end of constraints initialisation - have now set B[] and Z[] based on IsUsingConstraints, Constraints[] and DObs[]

        #Instrumentation block - mainly to start off the graph so you have something to look at while the next bit happens
        for (int k=0; k<NumModes; k++)
            InstrumentSetVariable("Beta"+k, Beta[k]);
        InstrumentSetVariable("delta", 0);
        InstrumentTimeInterval();
        #end of Instrumentation block

        FMatrix[] Tij = null;
        bool Converged = false;
        while (!Converged)
        {
            #Instrumentation block
            for (int k = 0; k < NumModes; k++)
            {
                InstrumentSetVariable("Beta" + k, Beta[k]);
                InstrumentSetVariable("delta" + k, 100);
                InstrumentSetVariable("delta", 100);
            }
            #end of instrumentation block
                

            bool ConstraintsMet = false;
            do
            {
                #residential constraints
                ConstraintsMet = true; //unless violated one or more times below
                int FailedConstraintsCount = 0;

                #model run
                Tij = new FMatrix[NumModes];
                for (int k = 0; k < NumModes; k++) //mode loop
                {
                    InstrumentStatusText = "Running model for mode "+k;
                    Tij[k] = new FMatrix(N, N);

                    Parallel.For(0, N, i =>
                    #for (int i = 0; i < N; i++)
                    {
                        #denominator calculation which is sum of all modes
                        double denom = 0;
                        for (int kk = 0; kk < NumModes; kk++) //second mode loop
                        {
                            for (int j = 0; j < N; j++)
                            {
                                denom += DjObs[j] * Math.Exp(-Beta[kk] * dis[kk]._M[i, j]);
                            }
                        }

                        #numerator calculation for this mode (k)
                        for (int j = 0; j < N; j++)
                        {
                            Tij[k]._M[i, j] = (float)(B[j] * OiObs[i] * DjObs[j] * Math.Exp(-Beta[k] * dis[k]._M[i, j]) / denom);
                        }
                    }
                    );
                }

                #constraints check
                if (IsUsingConstraints)
                {
                    System.Diagnostics.Debug.WriteLine("Constraints test");
                    InstrumentStatusText = "Constraints test";

                    for (int j = 0; j < N; j++)
                    {
                        float Dj = 0;
                        for (int i = 0; i < N; i++) Dj += Tij[0]._M[i, j]+Tij[1]._M[i,j]+Tij[2]._M[i,j];
                        if (Constraints[j] >= 1.0f) //Constraints is taking the place of Gj in the documentation
                        {
                            if ((Dj - Z[j]) >= 0.5) //was >1.0
                            {
                                B[j] = B[j] * Z[j] / Dj;
                                ConstraintsMet = false;
                                ++FailedConstraintsCount;
                                InstrumentStatusText = "Constraints violated on " + FailedConstraintsCount + " MSOA zones";
                                System.Diagnostics.Debug.WriteLine("Dj=" + Dj + " Zj=" + Z[j] + " Bj=" + B[j]);
                            }
                        }
                    }
                         
                    #copy B2 into B ready for the next round
                    #for (int j = 0; j < N; j++) B[j] = B2[j];
                }
                System.Diagnostics.Debug.WriteLine("FailedConstraintsCount=" + FailedConstraintsCount);

                #Instrumentation block
                #for (int k = 0; k < NumModes; k++)
                #    InstrumentSetVariable("Beta" + k, Beta[k]);
                #InstrumentSetVariable("delta", FailedConstraintsCount); //not technically delta, but I want to see it for testing
                #InstrumentTimeInterval();
                #end of instrumentation block

            } while (!ConstraintsMet);

            #calculate mean predicted trips and mean observed trips (this is CBar)
            float[] CBarPred = new float[NumModes];
            float[] CBarObs = new float[NumModes];
            float[] delta = new float[NumModes];
            for (int k = 0; k < NumModes; k++)
            {
                CBarPred[k] = CalculateCBar(ref Tij[k], ref dis[k]);
                CBarObs[k] = CalculateCBar(ref TObs[k], ref dis[k]);
                delta[k] = Math.Abs(CBarPred[k] - CBarObs[k]); //the aim is to minimise delta[0]+delta[1]+...
            }

            #delta check on all betas (Beta0, Beta1, Beta2) stopping condition for convergence
            #double gradient descent search on Beta0 and Beta1 and Beta2
            Converged = true;
            for (int k = 0; k < NumModes; k++)
            {
                if (delta[k] / CBarObs[k] > 0.001)
                {
                    Beta[k] = Beta[k] * CBarPred[k] / CBarObs[k];
                    Converged = false;
                }
            }
            #Instrumentation block
            for (int k = 0; k < NumModes; k++)
            {
                InstrumentSetVariable("Beta" + k, Beta[k]);
                InstrumentSetVariable("delta" + k, delta[k]);
            }
            InstrumentSetVariable("delta", delta[0] + delta[1] + delta[2]); //should be a k loop
            InstrumentTimeInterval();
            #end of instrumentation block
        }

        #Set the output, TPred[]
        for (int k = 0; k < NumModes; k++) TPred[k] = Tij[k];

        #debugging:
        #for (int i = 0; i < N; i++)
        #    System.Diagnostics.Debug.WriteLine("Quant3Model::Run::ConstraintsB," + i + "," + B[i]);

    }

###############################################################################

        /// <summary>
        /// NOTE: this was copied directly from the Quant 1 model
        /// Run the quant model with different values for the Oi and Dj zones.
        /// Uses a guid to store the user defined model run data
        /// PRE: needs TObs, dis and beta
        /// TODO: need to instrument this
        /// TODO: writes out one file, which is the sum of the three predicted matrices produced
        /// </summary>
        /// <param name="OiDjHash">Hashmap of zonei index and Oi, Dj values for that area. A value of -1 for Oi or Dj means no change.</param>
        /// <param name="hasConstraints">Run with random values added to the Dj values.</param>
        /// <param name="OutFilename">Filename of where to store the resulting matrix, probably includes the GUID of the user directory to store the results</param>
        public void RunWithChanges(Dictionary<int, float[]> OiDjHash, bool hasConstraints, string OutFilename)
        {
            Stopwatch timer = Stopwatch.StartNew();

            int N = TObs[0].N;

            float[] DjObs = new float[N];
            float[] OiObs = new float[N];
            float Sum;

            //OiObs
            for (int i = 0; i < N; i++)
            {
                Sum = 0;
                for (int j = 0; j < N; j++)
                {
                    for (int k = 0; k < NumModes; k++) Sum += TObs[k]._M[i, j];
                }
                OiObs[i] = Sum;
            }

            //DjObs
            for (int j = 0; j < N; j++)
            {
                Sum = 0;
                for (int i = 0; i < N; i++)
                {
                    for (int k = 0; k < NumModes; k++) Sum += TObs[k]._M[i, j];
                }
                DjObs[j] = Sum;
            }

            //this is a complete hack - generate a TPred matrix that we can get Dj constraints from
            FMatrix[] TPredCons = new FMatrix[NumModes];
            for (int k = 0; k < NumModes; k++) //mode loop
            {
                TPredCons[k] = new FMatrix(N, N);

                for (int i = 0; i < N; i++)
                {
                    //denominator calculation which is sum of all modes
                    double denom = 0;
                    for (int kk = 0; kk < NumModes; kk++) //second mode loop
                    {
                        for (int j = 0; j < N; j++)
                        {
                            denom += DjObs[j] * Math.Exp(-Beta[kk] * dis[kk]._M[i, j]);
                        }
                    }

                    //numerator calculation for this mode (k)
                    for (int j = 0; j < N; j++)
                    {
                        TPredCons[k]._M[i, j] = (float)(B[j] * OiObs[i] * DjObs[j] * Math.Exp(-Beta[k] * dis[k]._M[i, j]) / denom);
                    }
                }
            }
            //now the DjCons - you could just set Zj here?
            float[] DjCons = new float[N];
            for (int j = 0; j < N; j++)
            {
                Sum = 0;
                for (int i = 0; i < N; i++)
                {
                    for (int k = 0; k < NumModes; k++) Sum += TPredCons[k]._M[i, j];
                }
                DjCons[j] = Sum;
            }

            
            //
            //
            //TODO: Question - do the constraints take place before or after the Oi Dj changes? If before, then it's impossible to increase jobs in greenbelt zones. If after, then changes override the green belt.

            //constraints initialisation - this is the same as the calibration, except that the B[j] values are initially taken from the calibration, while Z[j] is initialised from Dj[j] as before.
            float[] Z = new float[N];
            for (int j = 0; j < N; j++)
            {
                Z[j] = float.MaxValue;
                if (IsUsingConstraints)
                {
                    if (Constraints[j] >= 1.0f) //constraints array taking place of Gj (Green Belt) in documentation
                    {
                        //Gj=1 means a high enough percentage of MSOA land is green belt, so can't be built on
                        //set constraint to original Dj
                        //Z[j] = DjObs[j];
                        Z[j] = DjCons[j];
                    }
                }
            }
            //end of constraints initialisation - have now set B[] and Z[] based on IsUsingConstraints, Constraints[] and DObs[]

            //apply changes here from the hashmap
            foreach (KeyValuePair<int, float[]> KVP in OiDjHash)
            {
                int i = KVP.Key;
                if (KVP.Value[0] >= 0) OiObs[i] = KVP.Value[0];
                if (KVP.Value[1] >= 0) DjObs[i] = KVP.Value[1];
            }


            bool ConstraintsMet = false;
            do
            {
                //residential constraints
                ConstraintsMet = true; //unless violated one or more times below
                int FailedConstraintsCount = 0;

                //run 3 model
                System.Diagnostics.Debug.WriteLine("Run 3 model");
                for (int k = 0; k < NumModes; k++) //mode loop
                {
                    TPred[k] = new FMatrix(N, N);

                    Parallel.For(0, N, i =>
                    //for (int i = 0; i < N; i++)
                    {
                        //denominator calculation which is sum of all modes
                        double denom = 0;
                        for (int kk = 0; kk < NumModes; kk++) //second mode loop
                        {
                            for (int j = 0; j < N; j++)
                            {
                                denom += B[j]*DjObs[j] * Math.Exp(-Beta[kk] * dis[kk]._M[i, j]);
                            }
                        }

                        //numerator calculation for this mode (k)
                        for (int j = 0; j < N; j++)
                        {
                            TPred[k]._M[i, j] = (float)(B[j] * OiObs[i] * DjObs[j] * Math.Exp(-Beta[k] * dis[k]._M[i, j]) / denom);
                        }
                    }
                    );
                }

                //constraints check
                if (IsUsingConstraints)
                {
                    System.Diagnostics.Debug.WriteLine("Constraints test");

                    for (int j = 0; j < N; j++)
                    {
                        float Dj = 0;
                        for (int i = 0; i < N; i++) Dj += TPred[0]._M[i, j] + TPred[1]._M[i, j] + TPred[2]._M[i, j];
                        if (Constraints[j] >= 1.0f) //Constraints is taking the place of Gj in the documentation
                        {
                            //System.Diagnostics.Debug.WriteLine("Test: " + Dj + ", " + Z[j] + "," + B[j]);
                            if ((Dj - Z[j]) >= 0.5) //was >1.0 threshold
                            {
                                B[j] = B[j] * Z[j] / Dj;
                                ConstraintsMet = false;
                                ++FailedConstraintsCount;
//                                System.Diagnostics.Debug.WriteLine("Constraints violated on " + FailedConstraintsCount + " MSOA zones");
//                                System.Diagnostics.Debug.WriteLine("Dj=" + Dj + " Zj=" + Z[j] + " Bj=" + B[j]);
                            }
                        }
                    }
                }
            } while (!ConstraintsMet);

            //add all three TPred together
            FMatrix TPredAll = new FMatrix(N, N);
            //Parallel.For(0, N, i =>
            for (int i = 0; i < N; i++)
            {
                for (int j = 0; j < N; j++)
                {
                    Sum = 0;
                    for (int k = 0; k < NumModes; k++)
                    {
                        Sum += TPred[k]._M[i, j];
                    }
                    TPredAll._M[i, j] = Sum;
                }
            }
            //);
            
            //and store the result somewhere
            TPredAll.DirtySerialise(OutFilename);

            System.Diagnostics.Debug.WriteLine("QUANT3Model::RunWithChanges: " + timer.ElapsedMilliseconds + " ms");
        }


###############################################################################
