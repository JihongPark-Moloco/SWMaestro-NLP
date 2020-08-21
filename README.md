# Keyword Extraction




## Algorithm
   
WordRank의 경우 충분한 데이터가 확보된 이후에 신뢰성 있는 추출이 가능하기에 영상이 최근에 업로드되어 그 댓글 양 자체가 부족할경우 신뢰도가 급격히 하락합니다.  
  
이를 보완하고자 각 채널에 카테고리를 선정, 해당 카테고리별로 키워드 사전을 이전에 존재하는 댓글이 충분하게 달려있는 영상들에서 추출하여
구축합니다.  
  
위를 이용해 새롭게 등록된 영상의 경우는 키워드 사전에서 영상의 제목과 설명을 비교해 키워드를 추출하고
이후 충분한 댓글이 누적될 경우 지속적으로 분석해 해당 영상의 키워드를 업데이트합니다.
### 키워드 추출
1. 수집된 영상의 댓글에 KR-WordRank를 적용, 키워드 후보를 구합니다.
2. 키워드 후보 중 영상의 제목과 설명에 포함되어있는 키워드를 선정합니다.  
이는 해당 영상의 키워드일 확률이 아주 높은 키워드로 판단되어집니다.
3. 키워드 후보들 중 제목과 설명에는 포함되어 있지 않지만 그 빈도수가 특출나게 높은등의 키워드일 확률이
아주 높은 후보들의 경우도 해당 영상의 키워드로 선정합니다.
### 키워드 사전 구축 및 활용
1. 카테고리 별로 나타나는 키워드를 취합하여 일정 빈도 이상의 키워드들로 이루어진 키워드사전을 구축합니다.  
2. 신규 영상의 경우 해당 영상에 적합한 카테고리의 키워드 사전에서 영상의 제목과 설명과 비교, 키워드 사전에 존재하는 키워드에 대해서는
그 즉시 해당 영상의 키워드로 선정합니다.
3. 이후 추가되어지는 댓글의 양이 일정치를 넘길 경우마다 추가로 분석을 시행, 해당 영상의 키워드를 추가합니다.


<!-- TABLE OF CONTENTS -->
## Table of Contents

* [About the Project](#about-the-project)
  * [Built With](#built-with)
* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
* [Usage](#usage)
* [Roadmap](#roadmap)
* [Contributing](#contributing)
* [License](#license)
* [Contact](#contact)
* [Acknowledgements](#acknowledgements)



<!-- ABOUT THE PROJECT -->
## About The Project
영상의 제목, 설명, 댓글을 통해 해당 영상의 키워드를 추출합니다.  
추출된 키워드를 통해 FastText 공간에서 채널들간의 거리를 계산해 유사 채널들을 찾아냅니다.  


### Built With
현 프로젝트는 다음의 주요 프레임워크를 통해 개발되었습니다.
* [KR-WorkRank](https://github.com/lovit/KR-WordRank)
* [Yake](https://github.com/LIAAD/yake)
* [FastText](https://github.com/facebookresearch/fastText)

### PipeLine
![image](https://13.125.91.162/swmaestro/muna-1/raw/master/images/NLP_pipeline.png)  

### Algorithm
#### Ko-WorkRank 
##### What is WordRank
WordRank 는 띄어쓰기가 없는 중국어와 일본어에서 graph ranking 알고리즘을 이용하여 단어를 추출하기 위해 제안된 방법입니다.  
Ranks 는 substring 의 단어 가능 점수이며, 이를 이용하여 unsupervised word segmentation 을 수행하였습니다.  
WordRank 는 substring graph 를 만든 뒤, graph ranking 알고리즘을 학습합니다.  
Substring graph 는 아래 그림의 (a), (b) 처럼 구성됩니다.  
먼저 문장에서 띄어쓰기가 포함되지 않은 모든 substring 의 빈도수를 계산합니다.  
이때 빈도수가 같으면서 짧은 substring 이 긴 substring 에 포함된다면 이를 제거합니다.  
아래 그림에서 ‘seet’ 의 빈도수가 2 이고, ‘seeth’ 의 빈도수가 2 이기 때문에 ‘seet’ 는 graph node 후보에서 제외됩니다.  
두번째 단계는 모든 substring nodes 에 대하여 links 를 구성합니다.  
‘that’ 옆에 ‘see’와 ‘dog’ 이 있었으므로 두 마디를 연결합니다.  
왼쪽에 위치한 subsrting 과 오른쪽에 위치한 subsrting 의 edge 는 서로 다른 종류로 표시합니다.  
이때, ‘do’ 역시 ‘that’의 오른쪽에 등장하였으므로 링크를 추가합니다.  
이렇게 구성된 subsrting graph 에 HITS 알고리즘을 적용하여 각 subsrting 의 ranking 을 계산합니다.  
![image](https://13.125.91.162/swmaestro/muna-1/raw/master/images/graph_wordrank_algorithm.png)  

##### Why WordRank 
유튜브의 경우에는 10대~30대 사이의 젊은 층의 점유율이 매우 높기에 은어와 신조어등에 대해 매우 민감하게 반응합니다.  
키워드 추출에서 지도 학습으로 접근할 경우 새롭게 파생되고 변형되어지는 모든 키워드들에 대응하기란 불가능합니다.  
그렇기에 비지도 학습 기반의 WordRank를 이용해 단어의 반복해서 나타나는 단어의 빈도수를 파악해 키워드를 추출합니다.  
WordRank를 이용할 시 통계에 기반하여 키워드를 추출하기에 사전 데이터 학습이 필요하지 않으며 새롭게 생겨나는 단어에 강인하고 
사용자가 실수로 발생시키는 오탈자등은 희석되어 전체 키워드 분석에서 제외되므로 키워드 추출 알고리즘으로 매우 적합합니다.  


현 프로젝트는 다음의 주요 프레임워크를 통해 개발되었습니다.
* [KR-WorkRank](https://github.com/lovit/KR-WordRank)
* [Yake](https://github.com/LIAAD/yake)
* [FastText](https://github.com/facebookresearch/fastText)



<!-- GETTING STARTED -->
## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### Prerequisites

This is an example of how to list things you need to use the software and how to install them.
* npm
```sh
npm install npm@latest -g
```

### Installation

1. Get a free API Key at [https://example.com](https://example.com)
2. Clone the repo
```sh
git clone https://github.com/your_username_/Project-Name.git
```
3. Install NPM packages
```sh
npm install
```
4. Enter your API in `config.js`
```JS
const API_KEY = 'ENTER YOUR API';
```



<!-- USAGE EXAMPLES -->
## Usage

Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.

_For more examples, please refer to the [Documentation](https://example.com)_



<!-- ROADMAP -->
## Roadmap

See the [open issues](https://github.com/othneildrew/Best-README-Template/issues) for a list of proposed features (and known issues).



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.



<!-- CONTACT -->
## Contact

Your Name - [@your_twitter](https://twitter.com/your_username) - email@example.com

Project Link: [https://github.com/your_username/repo_name](https://github.com/your_username/repo_name)



<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements
* [GitHub Emoji Cheat Sheet](https://www.webpagefx.com/tools/emoji-cheat-sheet)
* [Img Shields](https://shields.io)
* [Choose an Open Source License](https://choosealicense.com)
* [GitHub Pages](https://pages.github.com)
* [Animate.css](https://daneden.github.io/animate.css)
* [Loaders.css](https://connoratherton.com/loaders)
* [Slick Carousel](https://kenwheeler.github.io/slick)
* [Smooth Scroll](https://github.com/cferdinandi/smooth-scroll)
* [Sticky Kit](http://leafo.net/sticky-kit)
* [JVectorMap](http://jvectormap.com)
* [Font Awesome](https://fontawesome.com)





<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=flat-square
[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=flat-square
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=flat-square
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=flat-square
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=flat-square
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/othneildrew
[product-screenshot]: images/screenshot.png
 