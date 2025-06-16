<template>
    <div class="message-item">
        <!-- question -->
        <div class="question-container">
            <p class="question-text">{{ message }}</p>
        </div>

        <!-- answer -->
        <div v-if="answer !== null" class="answer-container">
            <p class="answer-text" v-html="compiledMarkdown"></p>
            <!-- icon tooltips-->
            <div class="icon-container">
                <span class="icon" :class="{ success: tooltips.copy === 'Copied!' }" :title="tooltips.copy"
                    @click="copyToClipboard"><svg width="15px" height="15px" viewBox="0 0 1024 1024"
                        xmlns="http://www.w3.org/2000/svg" fill="#000000" stroke="#000000" stroke-width="0.01024">
                        <g id="SVGRepo_bgCarrier" stroke-width="0"></g>
                        <g id="SVGRepo_tracerCarrier" stroke-linecap="round" stroke-linejoin="round"></g>
                        <g id="SVGRepo_iconCarrier">
                            <path fill="#6B6C6E"
                                d="M768 832a128 128 0 0 1-128 128H192A128 128 0 0 1 64 832V384a128 128 0 0 1 128-128v64a64 64 0 0 0-64 64v448a64 64 0 0 0 64 64h448a64 64 0 0 0 64-64h64z">
                            </path>
                            <path fill="#6B6C6E"
                                d="M384 128a64 64 0 0 0-64 64v448a64 64 0 0 0 64 64h448a64 64 0 0 0 64-64V192a64 64 0 0 0-64-64H384zm0-64h448a128 128 0 0 1 128 128v448a128 128 0 0 1-128 128H384a128 128 0 0 1-128-128V192A128 128 0 0 1 384 64z">
                            </path>
                        </g>
                    </svg></span>
                <!-- <span class="icon" :class="{ success: tooltips.thumbsup === 'Saved!' }" :title="tooltips.thumbsup"
                    @click="sendFeedback(1)"><svg width="15px" height="15px" viewBox="0 -2.5 64 64" version="1.1"
                        xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"
                        xmlns:sketch="http://www.bohemiancoding.com/sketch/ns" fill="#000000">
                        <g id="SVGRepo_bgCarrier" stroke-width="0"></g>
                        <g id="SVGRepo_tracerCarrier" stroke-linecap="round" stroke-linejoin="round"></g>
                        <g id="SVGRepo_iconCarrier">
                            <title>Thumb-up</title>
                            <desc>Created with Sketch.</desc>
                            <defs> </defs>
                            <g id="Page-1" stroke-width="3.2" fill="none" fill-rule="evenodd" sketch:type="MSPage">
                                <g id="Thumb-up" sketch:type="MSLayerGroup" transform="translate(1.000000, 2.000000)"
                                    stroke="#6B6C6E" stroke-width="3.2">
                                    <path
                                        d="M18,20 L2.9,20 C0.1,20 0,22.1 0,24.8 L0,49.1 C0,51.7 0.1,53.9 2.9,53.9 L18,53.9 C20.8,53.9 20.9,51.8 20.9,49.1 L20.9,24.8 C20.9,22.2 20.8,20 18,20 L18,20 Z"
                                        id="Shape" sketch:type="MSShapeGroup"> </path>
                                    <path
                                        d="M20.7,50.3 L22.1,50.3 C25.9,50.2 28.4,56 31.3,56 L53.9,56 C56.7,56 57.3,51.8 57.2,50 C57.2,50 60.7,48.4 59.1,42 C59.3,41.9 61.1,40.2 61.1,36.9 C61.1,33.6 59.2,32 59.2,32 C59.2,32 61.1,29.9 61.1,26.8 C61.1,23.7 58.4,22 55.6,22 L47.3,22 C40.1,22 40.7,18.1 40.7,18.1 C40.7,18.1 39.9,8 37.3,3.3 C34.1,-2.5 27.7,-0.6 30.1,7 C31.9,12.6 23.8,21 20.8,23.8"
                                        id="Shape" sketch:type="MSShapeGroup"> </path>
                                </g>
                            </g>
                        </g>
                    </svg></span>
                <span class="icon" :class="{ success: tooltips.thumbsdown === 'Saved!' }" :title="tooltips.thumbsdown"
                    @click="sendFeedback(0)"><svg width="15px" height="15px" viewBox="0 -2.5 64 64" version="1.1"
                        xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"
                        xmlns:sketch="http://www.bohemiancoding.com/sketch/ns" fill="#000000">
                        <g id="SVGRepo_bgCarrier" stroke-width="0"></g>
                        <g id="SVGRepo_tracerCarrier" stroke-linecap="round" stroke-linejoin="round"></g>
                        <g id="SVGRepo_iconCarrier">
                            <title>Thumb-down</title>
                            <desc>Created with Sketch.</desc>
                            <defs> </defs>
                            <g id="Page-1" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd"
                                sketch:type="MSPage">
                                <g id="Thumb-down" sketch:type="MSLayerGroup" transform="translate(2.000000, 1.000000)"
                                    stroke="#6B6C6E" stroke-width="3.2">
                                    <path
                                        d="M43,36 L58.1,36 C60.9,36 61,33.9 61,31.2 L61,6.9 C61,4.3 60.9,2.1 58.1,2.1 L43,2.1 C40.2,2.1 40.1,4.2 40.1,6.9 L40.1,31.2 C40.1,33.8 40.2,36 43,36 L43,36 Z"
                                        id="Shape" sketch:type="MSShapeGroup"> </path>
                                    <path
                                        d="M40.3,5.7 L38.9,5.7 C35.1,5.8 32.6,0 29.7,0 L7.1,0 C4.3,0 3.7,4.2 3.8,6 C3.8,6 0.3,7.6 1.9,14 C1.7,14.1 -0.1,15.8 -0.1,19.1 C-0.1,22.4 1.8,24 1.8,24 C1.8,24 -0.1,26.1 -0.1,29.2 C-0.1,32.3 2.6,34 5.4,34 L13.7,34 C20.9,34 20.3,37.9 20.3,37.9 C20.3,37.9 21.1,48 23.7,52.7 C26.9,58.5 33.3,56.6 30.9,49 C29.1,43.4 37.2,35 40.2,32.2"
                                        id="Shape" sketch:type="MSShapeGroup"> </path>
                                </g>
                            </g>
                        </g>
                    </svg>
                </span> -->
            </div>
        </div>

        <!-- processing dot -->
        <div v-else class="dots-loader">
            <div class="dot"></div>
            <div class="dot"></div>
            <div class="dot"></div>
        </div>
    </div>
</template>

<script setup>
import { computed, defineProps, ref } from 'vue';
import { marked } from 'marked';
import Prism from 'prismjs';
import '../assets/css/prism.css';

const props = defineProps({
    message: {
        type: String,
        required: true
    },
    answer: {
        type: String,
        default: ''
    },
    run_id: {
        type: String,
        default: ''
    }
});

const compiledMarkdown = computed(() => {
    return marked(props.answer, {
        gfm: true,
        breaks: true,
        headerIds: true,
        mangle: false,
        sanitize: false,
        smartLists: true,
        smartypants: true,
        xhtml: false,
        highlight: function (code, lang) {
            const languageClass = `language-${lang}`;
            if (Prism.languages[lang]) {
                return `<pre class="${languageClass} line-numbers"><code class="${languageClass}">${Prism.highlight(code, Prism.languages[lang], lang)}</code></pre>`;
            } else {
                return `<pre class="${languageClass} line-numbers"><code class="${languageClass}">${code}</code></pre>`;
            }
        }
    });
});

const tooltips = ref({
    copy: 'Copy',
    thumbsup: 'Helpful',
    thumbsdown: 'Not Helpful',
});

const copyToClipboard = () => {
    // 원본 마크다운 텍스트 사용
    navigator.clipboard.writeText(props.answer).then(() => {
        tooltips.value.copy = 'Copied!';

        setTimeout(() => {
            tooltips.value.copy = 'Copy';
        }, 2000);
    }).catch(err => {
        console.error('Failed to copy: ', err);
    });
};

// const isFeedbackSent = ref(false); // 상태 변수 추가

// Function to send feedback to the server
// const sendFeedback = async (score) => {
//     if (isFeedbackSent.value) return; // 이미 피드백을 보냈으면 함수 종료

//     try {
//         const response = await fetch(`${import.meta.env.VITE_API_URL}/feedback`, {
//             method: 'POST',
//             credentials: 'include',
//             headers: {
//                 'Content-Type': 'application/json',
//             },
//             body: JSON.stringify({
//                 score,
//                 run_id: props.run_id
//             }),
//         });

//         if (!response.ok) {
//             throw new Error('Failed to send feedback');
//         }

//         // Change tooltip text and color to 'Saved!'
//         tooltips.value[score === 1 ? 'thumbsup' : 'thumbsdown'] = 'Saved!';
//         isFeedbackSent.value = true; // 피드백이 전송되었음을 표시

//         // Revert tooltip text and color back after 2 seconds
//         setTimeout(() => {
//             tooltips.value[score === 1 ? 'thumbsup' : 'thumbsdown'] = score === 1 ? 'Helpful' : 'Not Helpful';
//         }, 2000);

//         console.log('Feedback sent successfully');
//     } catch (error) {
//         console.error('Error sending feedback:', error);
//     }
// };

</script>

<style scoped>
@import '../assets/css/prism.css';
@import '../assets/css/main.css';
</style>