<!-- app.vue -->
<template>
  <NuxtPage />
</template>

<script setup>
import { useRouter, useRoute } from '#app'

const router = useRouter()
const route = useRoute()

// 초기 로딩 시 리디렉션
// if (route.path === '/') {
//   router.push('/login')
// }
</script>