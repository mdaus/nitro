#include <mem/ScratchMemory.h>
#include <mem/BufferView.h>
#include <cli/ArgumentParser.h>
#include <str/Convert.h>
#include <vector>
#include <fstream>
#include <string>

struct Operation
{
    Operation(const std::string op,
              const std::string name,
              const size_t bytes) :
        op(op),
        name(name),
        bytes(bytes)
    {
    }

    std::string op;
    std::string name;
    size_t bytes;
    mem::BufferView<sys::ubyte> buffer;
};

/*!
 * \class Visualizer
 *
 * \brief Handles the inclusion and visualization of a put or release operation
 */
class Visualizer
{
public:
    /*!
     * Constructor
     *
     * \param prevOperations All operations handled in previous iterations
     * \param iteration Which iteration is being processed
     * \param[in,out] htmlFile Ofstream to .html file
     * \param[in,out] cssFile Ofstream to .css file
     */
    Visualizer(const std::vector<Operation> prevOperations,
               const size_t iteration,
               std::ofstream& htmlFile,
               std::ofstream& cssFile):
        mPrevOperations(prevOperations),
        mIteration(iteration),
        mHTMLFile(htmlFile),
        mCSSFile(cssFile)
    {
        mStartPtr = NULL;
        mColors.push_back("lightgrey");
        mColors.push_back("lightblue");
        mColors.push_back("lightcyan");
        mReleasedColors.push_back("indianred");
        mReleasedColors.push_back("coral");
        mReleasedColors.push_back("salmon");
    }

    /*!
     * Find the lowest start address, which dictates the first block, and set mStartPtr.
     *
     * \param operations A vector of all operations
     */
    void set_start_ptr(const std::vector<Operation>& operations)
    {
        mStartPtr = operations.at(0).buffer.data;
        for (size_t ii = 0; ii < operations.size(); ++ii)
        {
            if (operations.at(ii).buffer.data - mStartPtr <= 0)
            {
                mStartPtr = operations.at(ii).buffer.data;
            }
        }
    }

    /*!
     * Write the CSS and HTML to create a box for a single operation.
     *
     * \param op Operation to draw a box of
     * \param color_iter Iterator to arbitrarily select a color
     */
    void create_box(const Operation op, const size_t color_iter)
    {
        std::string color = mColors.at(color_iter % 3);
        std::string height = "25px;\n";
        if (op.op == "release")
        {
            color = mReleasedColors.at(color_iter % 3);
            height = "30px;\n";
        }

        mCSSFile << "#" << op.name << " { \n";
        mCSSFile << "height: " << height;
        mCSSFile << "display: inline-block;\n";
        mCSSFile << "position: absolute;\n";
        mCSSFile << "background-color: " << color << ";\n";
        mCSSFile << "width: " << op.buffer.size << "px;\n";
        mCSSFile << "margin-left: " << op.buffer.data - mStartPtr << "px;\n";
        mCSSFile << "border-style: solid;\n";
        mCSSFile << "}\n";

        mHTMLFile << "<div id=\"" << op.name << "\">&nbsp;" << op.name[0] << "</div>\n";
    }

    /*!
     * When building a new visualizer, all operations that have already been done
     * in previous iterations must be recreated.
     *
     * \param[out] currentOperations Operations for this iteration
     * \param[out] scratch Scratch memory object
     */
    void handlePrevOps(std::vector<Operation>& currentOperations, mem::ScratchMemory& scratch)
    {
        for (size_t ii = 0; ii < mPrevOperations.size(); ++ii)
        {
            const std::string segmentName = std::string(1, mPrevOperations.at(ii).name[0]) +
                    str::toString(mIteration);

            if (mPrevOperations.at(ii).op == "put")
            {
                scratch.put<sys::ubyte>(segmentName, mPrevOperations.at(ii).bytes, 1, 1);
                currentOperations.push_back(Operation("put",
                                                      segmentName,
                                                      mPrevOperations.at(ii).bytes));
            }
            else if (mPrevOperations.at(ii).op == "release")
            {
                scratch.release(segmentName);
                currentOperations.push_back(Operation("release",
                                                      segmentName,
                                                      mPrevOperations.at(ii).bytes));
            }
        }
    }

    /*!
     * Runs the randomly generated test case.
     *
     * \param[in,out] currentOperations Operations for this iteration
     * \param[in,out] bufferName The name/key of the new segment
     * \param[in,out] notReleasedKeys The keys that have not been released
     * \param[out] usedBufferSpace How much memory has already been used up
     * \param[out] scratch Scratch memory object
     */
    void randomTest(std::vector<Operation>& currentOperations,
                    unsigned char& bufferName,
                    std::vector<unsigned char>& notReleasedKeys,
                    size_t& usedBufferSpace,
                    mem::ScratchMemory& scratch)
    {
        handlePrevOps(currentOperations, scratch);

        const size_t numElements = (rand() % 150) + 20;
        unsigned int releaseIfThree = (rand() % 3) + 1;

        if ((releaseIfThree == 3) && (currentOperations.size() > 1) && !notReleasedKeys.empty())
        {
            unsigned int keyToReleaseIndex = (rand() % notReleasedKeys.size());
            const std::string segmentName = std::string(1, notReleasedKeys.at(keyToReleaseIndex)) +
                    str::toString(mIteration);

            scratch.release(segmentName);
            notReleasedKeys.erase(notReleasedKeys.begin() + keyToReleaseIndex);

            currentOperations.push_back(Operation("release", segmentName, numElements));
        }
        else
        {
            const std::string segmentName = std::string(1, bufferName) + str::toString(mIteration);
            scratch.put<sys::ubyte>(segmentName, numElements, 1, 1);
            currentOperations.push_back(Operation("put", segmentName, numElements));
            notReleasedKeys.push_back(bufferName);

            ++bufferName;
            usedBufferSpace += numElements;
        }
    }

    /*!
     * Runs a version of the unittest in which concurrent keys are released in a very specific order
     *
     * \param[in,out] currentOperations Operations for this iteration
     * \param[in,out] bufferName The name/key of the new segment
     * \param[in,out] notReleasedKeys The keys that have not been released
     * \param[in,out] testIter Used to keep track of which step of the unittest to do
     * \param[out] usedBufferSpace How much memory has already been used up
     * \param[out] scratch Scratch memory object
     */
    void concurrentBlockTest(std::vector<Operation>& currentOperations,
                             unsigned char& bufferName,
                             size_t& testIter,
                             size_t& usedBufferSpace,
                             mem::ScratchMemory& scratch)
    {
        handlePrevOps(currentOperations, scratch);
        size_t numElements = (rand() % 150) + 20;

        if (testIter == 0)
        {
            const std::string segmentName = std::string(1, bufferName) +
                    str::toString(mIteration);
            scratch.put<sys::ubyte>(segmentName, numElements, 1, 1);
            currentOperations.push_back(Operation("put", segmentName, numElements));
            ++testIter;
            ++bufferName;
        }
        else if (testIter == 1)
        {
            const std::string segmentName = std::string(1, bufferName) +
                    str::toString(mIteration);
            scratch.put<sys::ubyte>(segmentName, numElements, 1, 1);
            currentOperations.push_back(Operation("put", segmentName, numElements));
            ++testIter;
            ++bufferName;
        }
        else if (testIter == 2)
        {
            const std::string segmentName = std::string(1, bufferName) +
                    str::toString(mIteration);
            scratch.put<sys::ubyte>(segmentName, numElements, 1, 1);
            currentOperations.push_back(Operation("put", segmentName, numElements));
            ++testIter;
            ++bufferName;
        }
        else if (testIter == 3)
        {
            const std::string segmentName = std::string(1, bufferName - 2) +
                    str::toString(mIteration);
            scratch.release(segmentName);
            currentOperations.push_back(Operation("release", segmentName, numElements));
            ++testIter;
        }
        else if (testIter == 4)
        {
            const std::string segmentName = std::string(1, bufferName - 1) +
                    str::toString(mIteration);
            scratch.release(segmentName);
            currentOperations.push_back(Operation("release", segmentName, numElements));
            ++testIter;
        }
        else if (testIter == 5)
        {
            const std::string segmentName = std::string(1, bufferName) +
                    str::toString(mIteration);
            scratch.put<sys::ubyte>(segmentName, numElements, 1, 1);
            currentOperations.push_back(Operation("put", segmentName, numElements));
            ++testIter;
            ++bufferName;
        }
        else if (testIter == 6)
        {
            const std::string segmentName = std::string(1, bufferName - 4) +
                    str::toString(mIteration);
            scratch.release(segmentName);
            currentOperations.push_back(Operation("release", segmentName, numElements));
            testIter = 0;
        }
    }

    /*!
     * Runs a version of the unittest in which connected keys are released in a very specific order
     *
     * \param[in,out] currentOperations Operations for this iteration
     * \param[in,out] bufferName The name/key of the new segment
     * \param[in,out] notReleasedKeys The keys that have not been released
     * \param[in,out] testIter Used to keep track of which step of the unittest to do
     * \param[out] usedBufferSpace How much memory has already been used up
     * \param[out] scratch Scratch memory object
     */
    void connectedBlockTest(std::vector<Operation>& currentOperations,
                             unsigned char& bufferName,
                             size_t& testIter,
                             size_t& usedBufferSpace,
                             mem::ScratchMemory& scratch)
    {
        handlePrevOps(currentOperations, scratch);
        size_t numElements = (rand() % 150) + 20;

        if (testIter == 0)
        {
            const std::string segmentName = std::string(1, bufferName) +
                    str::toString(mIteration);
            scratch.put<sys::ubyte>(segmentName, numElements, 1, 1);
            currentOperations.push_back(Operation("put", segmentName, numElements));
            ++testIter;
            ++bufferName;
        }
        else if (testIter == 1)
        {
            const std::string segmentName = std::string(1, bufferName) +
                    str::toString(mIteration);
            scratch.put<sys::ubyte>(segmentName, numElements, 1, 1);
            currentOperations.push_back(Operation("put", segmentName, numElements));
            ++testIter;
            ++bufferName;
        }
        else if (testIter == 2)
        {
            const std::string segmentName = std::string(1, bufferName - 1) +
                    str::toString(mIteration);
            scratch.release(segmentName);
            currentOperations.push_back(Operation("release", segmentName, numElements));
            ++testIter;
        }
        else if (testIter == 3)
        {
            const std::string segmentName = std::string(1, bufferName) +
                    str::toString(mIteration);
            scratch.put<sys::ubyte>(segmentName, numElements, 1, 1);
            currentOperations.push_back(Operation("put", segmentName, numElements));
            ++testIter;
            ++bufferName;
        }
        else if (testIter == 4)
        {
            const std::string segmentName = std::string(1, bufferName - 3) +
                    str::toString(mIteration);
            scratch.release(segmentName);
            currentOperations.push_back(Operation("release", segmentName, numElements));
            ++testIter;
        }
        else if (testIter == 5)
        {
            const std::string segmentName = std::string(1, bufferName) +
                    str::toString(mIteration);
            scratch.put<sys::ubyte>(segmentName, numElements, 1, 1);
            currentOperations.push_back(Operation("put", segmentName, numElements));
            ++testIter;
            ++bufferName;
        }
        else if (testIter == 6)
        {
            const std::string segmentName = std::string(1, bufferName - 2) +
                    str::toString(mIteration);
            scratch.release(segmentName);
            currentOperations.push_back(Operation("release", segmentName, numElements));
            testIter = 0;
        }
    }

private:
    std::vector<std::string> mColors;
    std::vector<std::string> mReleasedColors;
    std::vector<Operation> mPrevOperations;
    sys::ubyte* mStartPtr;
    size_t mIteration;

    std::ofstream& mHTMLFile;
    std::ofstream& mCSSFile;
};

int main(int argc, char** argv)
{
    cli::ArgumentParser parser;

    parser.setDescription("Software to visualize scratch memory test cases in HTML/CSS");
    parser.addArgument("--test", "Select which test case to run", cli::STORE, "test")->setDefault("random");
    parser.addArgument("--numBytes", "Determines how much memory is allocated for the buffer",
                       cli::STORE, "bytes", "INT")->setDefault(1000);

    const cli::Results* options(parser.parse(argc, argv));
    const std::string testType(options->get<std::string>("test"));

    srand((unsigned)time(0));

    std::ofstream htmlFile;
    htmlFile.open("scratch_release.html");
    std::ofstream cssFile;
    cssFile.open("style.css");

    //Headers
    htmlFile << "<!DOCTYPE HTML>\n";
    htmlFile << "<html>\n";
    htmlFile << "<head>\n";
    htmlFile << "<link rel=\"stylesheet\" type=\"text/css\" href=\"style.css\">\n";
    htmlFile << "</head>\n";
    htmlFile << "<body>\n";

    std::vector<Operation> prevOperations;
    unsigned char bufferName = 'a';

    const size_t totalBufferSpace(options->get<int>("bytes"));
    size_t usedBufferSpace = 0;
    size_t testIter = 0;

    std::vector<unsigned char> notReleasedKeys;

    for (size_t jj = 0; jj < 20; ++jj)
    {
        std::vector<Operation> currentOperations;
        mem::ScratchMemory scratch;
        Visualizer visualize(prevOperations, jj, htmlFile, cssFile);

        if (testType == "random")
        {   visualize.randomTest(currentOperations,
                                 bufferName,
                                 notReleasedKeys,
                                 usedBufferSpace,
                                 scratch);
        }
        else if (testType == "concurrent")
        {
            visualize.concurrentBlockTest(currentOperations,
                                          bufferName,
                                          testIter,
                                          usedBufferSpace,
                                          scratch);
        }
        else if (testType == "connected")
        {
            visualize.connectedBlockTest(currentOperations,
                                         bufferName,
                                         testIter,
                                         usedBufferSpace,
                                         scratch);
        }
        else
        {
            std::cout << "--test must be \"random\", \"concurrent\", or \"connected\"\n";
            return 1;
        }

        std::vector<sys::ubyte> storage(totalBufferSpace);
        mem::BufferView<sys::ubyte> buffer(storage.data(), storage.size());

        if (usedBufferSpace > totalBufferSpace)
        {
            break;
        }
        scratch.setup(buffer);

        //This draws the line. Draw one per scratch instance
        htmlFile << "<hr><br>\n";
        cssFile << "hr { \n";
        cssFile << "height: 1px;\n";
        cssFile << "float: left;\n";
        cssFile << "width: " << buffer.size << "px;\n";
        cssFile << "position: absolute;\n";
        cssFile << "}\n";

        //Goes through each buffer in order
        for (size_t ii = 0; ii < currentOperations.size(); ++ii)
        {
            currentOperations.at(ii).buffer =
                    scratch.getBufferView<sys::ubyte>(currentOperations.at(ii).name);
        }

        visualize.set_start_ptr(currentOperations);

        for (size_t ii = 0; ii < currentOperations.size(); ++ii)
        {
                visualize.create_box(currentOperations.at(ii), ii);
        }

        htmlFile << "<br><br><br>\n";
        prevOperations = currentOperations;
    }


    htmlFile << "</body>\n";
    htmlFile << "</html>\n";

    htmlFile.close();
    cssFile.close();

    try
    {
        system("firefox scratch_release.html");
    }
    catch(except::Exception& ex)
    {
        std::cout << "Failed to open html file in firefox\n";
    }

    return 0;
}
