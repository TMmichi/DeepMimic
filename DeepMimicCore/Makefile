BIN=Main
OBJ_DIR = objs
MKDIR = mkdir -p

#EIGEN_DIR = ../libraries/eigen-3.3.8
#BULLET_INC_DIR = ../libraries/bullet3-2.88/src
#PYTHON_INC = /home/mlcs/anaconda3/envs/thesis/include/python3.6m
#PYTHON_LIB = /home/mlcs/anaconda3/envs/thesis/bin/ -lpython3.6m

EIGEN_DIR = ../libraries/eigen-3.3.7
BULLET_INC_DIR = ../libraries/bullet3-2.88/src
PYTHON_INC = /home/ljh/anaconda3/envs/mujaco/include/python3.6m
PYTHON_LIB = /home/ljh/anaconda3/envs/mujaco/bin/ -lpython3.6m

INC = -I./ \
	-I$(EIGEN_DIR) \
	-I$(BULLET_INC_DIR)

LIBS = -lGLEW -lGL -lGLU -lglut -lBulletDynamics -lBulletCollision -lLinearMath -lm -lstdc++

CC = clang++
CFLAGS = -std=c++11 -O3 -Wall -fPIC 

SRC_DIRS = util/ \
	util/json/ \
	anim/ \
	sim/ \
	render/ \
	render/lodepng/ \
	scenes/ \
	scenes/arm/ \
	scenes/particle/ \
	scenes/pendulum/

SRC_FILES = $(BIN).cpp DeepMimicCore.cpp $(foreach dir, $(SRC_DIRS), $(wildcard $(dir)*.cpp))
OUT_DIRS = $(addprefix $(OBJ_DIR)/, $(SRC_DIRS))
OBJ = $(addprefix $(OBJ_DIR)/, $(patsubst %.cpp, %.o, $(SRC_FILES)))


default: build

build: $(BIN)

print:
	$(OUT_DIRS)

python: $(OUT_DIRS) $(OBJ)
	swig -c++ -python DeepMimicCore.i
	$(CC) -c -g $(CFLAGS) $(INC) $(LIBS) DeepMimicCore.cpp DeepMimicCore_wrap.cxx -I$(PYTHON_INC)
	$(CC) -shared $(CFLAGS) $(OBJ) $(LIBS) DeepMimicCore_wrap.o -o _DeepMimicCore.so -L$(PYTHON_LIB)

$(OBJ_DIR)/%.o: %.cpp
	$(CC) -c -g $(CFLAGS) $(INC) -o $@ $<

$(BIN): $(OUT_DIRS) $(OBJ)
	$(CC) -g $(CFLAGS) $(OBJ) $(LIBS) -o $@

$(OUT_DIRS):
	$(MKDIR) $(OUT_DIRS)
	
clean:
	- rm -f -r $(OUT_DIRS) $(BIN) $(OBJ) $(OBJ_DIR) DeepMimicCore_wrap.cxx _DeepMimicCore.so
