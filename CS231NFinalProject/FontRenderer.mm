//
//  FontRenderer.m
//  CS231NFinalProject
//
//  Created by Luc Rosenzweig on 6/3/23.
//

#if 1
#import <Foundation/Foundation.h>

#import "FontRenderer.h"
#import <MetalKit/MetalKit.h>

#define FONT_NAME @"fixedsys"

struct FontVertex
{
    vector_float2 position;
    vector_float2 uvs;
};

struct BoxVertex
{
    vector_float4 position;
    vector_float4 color;
};

struct FontCharacter
{
    char characterValue;
    vector_float2 uvsBase;
    vector_float2 uvsSize;
    vector_float2 displaySize;
    vector_float2 offset;
    float advance;
};

struct FontWord
{
    char *pointer;
    uint16_t size;
};

// Except new line
static char *fontSkipBreakCharacters(char *pointer) {
    for (;;) {
        switch (*pointer) {
        case ' ': case '=': ++pointer;
        default: return pointer;
        }
    }
    return NULL;
}

static char *fontGotoNextBreakCharacter(
    char *pointer) {
    for (;;) {
        switch (*pointer) {
        case ' ': case '=': case '\n': return pointer;
        default: ++pointer;
        }
    }
    return NULL;
}

static char *fontSkipLine(
    char *pointer) {
    for (;;) {
        switch (*pointer) {
        case '\n': return ++pointer;
        default: ++pointer;
        }
    }
    return NULL;
}

static char *fontMoveAndGetWord(
    char *pointer,
    FontWord *dst_word) {
    char *new_pointer = fontGotoNextBreakCharacter(pointer);
    if (dst_word) {
        dst_word->pointer = pointer;
        dst_word->size = (int16_t)(new_pointer - pointer);
    }
    return(new_pointer);
}

static char *fontSkipUntilDigit(
    char *pointer) {
    for (;;) {
        switch (*pointer) {
        case '-': case '0': case '1': case '2': case '3': case '4':
        case '5': case '6': case '7': case '8': case '9': return pointer;
        default: ++pointer;
        }
    }
    return NULL;
}

struct FontStringNumber
{
    FontWord word;
    char characters[10];
};

static int32_t fontAtoi(
    FontWord word) {
    FontStringNumber number;
    memcpy(number.characters, word.pointer, sizeof(char) * word.size);
    number.characters[word.size] = '\0';
    return(atoi(number.characters));
}

static char *fontGetCharCount(
    char *pointer,
    int32_t *count) {
    for (;;) {
        FontWord charsStr;
        pointer = fontMoveAndGetWord(pointer, &charsStr);
        if (charsStr.size == strlen("chars")) {
            pointer = fontSkipUntilDigit(pointer);
            FontWord countStr;
            pointer = fontMoveAndGetWord(pointer, &countStr);
            *count = fontAtoi(countStr);
            pointer = fontSkipLine(pointer);
            return pointer;
        }
        pointer = fontSkipLine(pointer);
    }
}

static char *fontGetFontAttributeValue(
    char *pointer,
    int32_t *value) {
    pointer = fontSkipUntilDigit(pointer);
    FontWord valueStr;
    pointer = fontMoveAndGetWord(pointer, &valueStr);
    *value = fontAtoi(valueStr);
    return(pointer);
}

// Returns char count
static int loadFont(FontCharacter *dstChars) {
    static const float FONT_MAP_W = 512.0f, FONT_MAP_H = 512.0f;
    
    NSString *path = [[NSBundle mainBundle] pathForResource:FONT_NAME ofType:@"fnt" inDirectory:@""];
    NSString *contents = [NSString stringWithContentsOfFile:path encoding:NSASCIIStringEncoding error:nil];

    const char *fntContents = [contents cStringUsingEncoding:NSASCIIStringEncoding];

    int charCount = 0;
    char *currentChar = fontGetCharCount((char *)fntContents, &charCount);

    // Ready to start parsing the file
    for (uint32_t i = 0; i < (uint32_t)charCount; ++i) {
        // Char ID value
        int32_t charId = 0;
        currentChar = fontGetFontAttributeValue(currentChar, &charId);

        // X value
        int32_t x = 0;
        currentChar = fontGetFontAttributeValue(currentChar, &x);

        // X value
        int32_t y = 0;
        currentChar = fontGetFontAttributeValue(currentChar, &y);
        y = (int32_t)FONT_MAP_H - y;
        
        // Width value
        int32_t width = 0;
        currentChar = fontGetFontAttributeValue(currentChar, &width);

        // Height value
        int32_t height = 0;
        currentChar = fontGetFontAttributeValue(currentChar, &height);

        // XOffset value
        int32_t xoffset = 0;
        currentChar = fontGetFontAttributeValue(currentChar, &xoffset);

        // YOffset value
        int32_t yoffset = 0;
        currentChar = fontGetFontAttributeValue(currentChar, &yoffset);

        // XAdvanc value
        int32_t xadvance = 0;
        currentChar = fontGetFontAttributeValue(currentChar, &xadvance);

        FontCharacter *character = &dstChars[charId];
        character->characterValue = (char)charId;
        // ----------------------------------------------------------------------------- \/ Do y - height so that base position is at bottom of character
        character->uvsBase = simd_make_float2((float)x / (float)FONT_MAP_W, (float)(y - height) / (float)FONT_MAP_H);
        character->uvsSize = simd_make_float2((float)width / (float)FONT_MAP_W, (float)height / (float)FONT_MAP_H);
        character->displaySize = simd_make_float2((float)width / (float)xadvance, (float)height / (float)xadvance);
        character->offset = simd_make_float2((float)xoffset / (float)xadvance, (float)yoffset / (float)xadvance);
        character->offset.y *= -1.0f;
        character->advance = (float)xadvance / (float)xadvance;
        
        currentChar = fontSkipLine(currentChar);
    }
    
    return charCount;
}

struct Text
{
    static const uint32_t MAX_CHARS = 64;
    
    uint32_t colors[MAX_CHARS] = {};
    char characters[MAX_CHARS] = {};
    uint32_t charCount = 0;
    
    // This is in 0-1 coordinates in all directions.
    vector_float2 coordStart;
};

@implementation FontRenderer
{
    id<MTLTexture> mFontAtlas;
    
    int mCharCount;
    FontCharacter mCharacters[0xFF];
    
    id<MTLRenderPipelineState> mTextPipeline;
    id<MTLRenderPipelineState> mBoundingBoxPipeline;
}

-(nonnull instancetype)initWithDevice:(id<MTLDevice>)device defaultLibrary:(id<MTLLibrary>)defaultLib view:(MTKView *)view
{
    self = [super init];
    
    if (self)
    {
        NSError *error = nil;
        
        // Load the texture file
        MTKTextureLoader *loader = [[MTKTextureLoader alloc] initWithDevice: device];
        
        NSString* path = [[NSBundle mainBundle] pathForResource:FONT_NAME ofType:@"png" inDirectory:@""];
        NSURL *url = [NSURL fileURLWithPath:path];
        mFontAtlas = [loader newTextureWithContentsOfURL:url options:nil error:&error];
        
        if (error)
            NSLog(@"%@", error.description);
        
        mCharCount = loadFont(mCharacters);
        
        {
            // Create text rendering pipeline state.
            MTLVertexDescriptor *vtxDesc = [[MTLVertexDescriptor alloc] init];
            vtxDesc.attributes[0].format = MTLVertexFormatFloat2;
            vtxDesc.attributes[0].offset = 0;
            vtxDesc.attributes[0].bufferIndex = 0;
            vtxDesc.attributes[1].format = MTLVertexFormatFloat2;
            vtxDesc.attributes[1].offset = sizeof(float)*2;
            vtxDesc.attributes[1].bufferIndex = 0;
            vtxDesc.layouts[0].stride = sizeof(float)*4;
            vtxDesc.layouts[0].stepRate = 1;
            vtxDesc.layouts[0].stepFunction = MTLVertexStepFunctionPerVertex;
            
            id<MTLFunction> vertexFunction = [defaultLib newFunctionWithName:@"renderFontVertex"];
            id<MTLFunction> fragmentFunction = [defaultLib newFunctionWithName:@"renderFontFragment"];
            
            MTLRenderPipelineDescriptor *pipelineStateDescriptor = [[MTLRenderPipelineDescriptor alloc] init];
            pipelineStateDescriptor.label = @"FontRendererPipeline";
            pipelineStateDescriptor.rasterSampleCount = 1;
            pipelineStateDescriptor.vertexFunction = vertexFunction;
            pipelineStateDescriptor.fragmentFunction = fragmentFunction;
            pipelineStateDescriptor.vertexDescriptor = vtxDesc;
            pipelineStateDescriptor.colorAttachments[0].pixelFormat = view.colorPixelFormat;
            pipelineStateDescriptor.depthAttachmentPixelFormat = view.depthStencilPixelFormat;
            pipelineStateDescriptor.stencilAttachmentPixelFormat = view.depthStencilPixelFormat;
            
            mTextPipeline = [device newRenderPipelineStateWithDescriptor:pipelineStateDescriptor error:&error];
            if (!mTextPipeline)
            {
                NSLog(@"Failed to created pipeline state, error %@", error);
            }
        }
        
        {
            MTLVertexDescriptor *vtxDesc = [[MTLVertexDescriptor alloc] init];
            vtxDesc.attributes[0].format = MTLVertexFormatFloat4;
            vtxDesc.attributes[0].offset = 0;
            vtxDesc.attributes[0].bufferIndex = 0;
            vtxDesc.attributes[1].format = MTLVertexFormatFloat4;
            vtxDesc.attributes[1].offset = sizeof(float)*4;
            vtxDesc.attributes[1].bufferIndex = 0;
            
            vtxDesc.layouts[0].stride = sizeof(float)*8;
            vtxDesc.layouts[0].stepRate = 1;
            vtxDesc.layouts[0].stepFunction = MTLVertexStepFunctionPerVertex;
            
            id<MTLFunction> vertexFunction = [defaultLib newFunctionWithName:@"renderBoxVertex"];
            id<MTLFunction> fragmentFunction = [defaultLib newFunctionWithName:@"renderBoxFragment"];
            
            MTLRenderPipelineDescriptor *boxPipelineStateDescriptor = [[MTLRenderPipelineDescriptor alloc] init];
            boxPipelineStateDescriptor.label = @"BoxRendererPipeline";
            boxPipelineStateDescriptor.rasterSampleCount = 1;
            boxPipelineStateDescriptor.vertexFunction = vertexFunction;
            boxPipelineStateDescriptor.fragmentFunction = fragmentFunction;
            boxPipelineStateDescriptor.vertexDescriptor = vtxDesc;
            boxPipelineStateDescriptor.colorAttachments[0].pixelFormat = view.colorPixelFormat;
            boxPipelineStateDescriptor.depthAttachmentPixelFormat = view.depthStencilPixelFormat;
            boxPipelineStateDescriptor.stencilAttachmentPixelFormat = view.depthStencilPixelFormat;
            
            mBoundingBoxPipeline = [device newRenderPipelineStateWithDescriptor:boxPipelineStateDescriptor error:&error];
            if (!mBoundingBoxPipeline)
            {
                NSLog(@"Failed to created pipeline state, error %@", error);
            }
        }
    }
    
    return self;
}

#define MAX_BOX_BUFFER_SIZE (sizeof(BoxVertex) * 8 * 500)

-(struct BoxRenderInfo)makeBoxRenderInfo:(nonnull id<MTLDevice>)device
{
    struct BoxRenderInfo renderInfo;
    renderInfo.vertexCount = 0;
    renderInfo.vertexBuffer = [device newBufferWithLength:MAX_BOX_BUFFER_SIZE options:MTLResourceStorageModeShared];
    return renderInfo;
}

-(void)pushBoxPixelCoords:(nonnull struct BoxRenderInfo *)info position:(vector_int2)pos size:(vector_int2)size color:(vector_float4)color viewport:(vector_uint2)viewport
{
    // Get NDC position
    vector_float2 ndcPos = simd_make_float2((float)pos.x / (float)viewport.x, (float)pos.y / (float)viewport.y);
    ndcPos *= 2.0f;
    ndcPos -= simd_make_float2(1.0f, 1.0f);
    
    // Get NDC size
    vector_float2 ndcSize = simd_make_float2((float)size.x / (float)viewport.x, (float)size.y / (float)viewport.y);
    ndcSize *= 2.0f;
    
    // Push vertices
    BoxVertex *vertices = (BoxVertex *)[info->vertexBuffer contents];
    
    vector_float4 ndcPosV4 = simd_make_float4(ndcPos.x, ndcPos.y, 0.0f, 1.0f);
    
    vertices[info->vertexCount++] = { ndcPosV4, color };
    vertices[info->vertexCount++] = { ndcPosV4 + simd_make_float4(ndcSize.x, 0.0f, 0.0f, 0.0f), color };
    
    vertices[info->vertexCount++] = { ndcPosV4 + simd_make_float4(ndcSize.x, 0.0f, 0.0f, 0.0f), color };
    vertices[info->vertexCount++] = { ndcPosV4 + simd_make_float4(ndcSize.x, ndcSize.y, 0.0f, 0.0f), color };
    
    vertices[info->vertexCount++] = { ndcPosV4 + simd_make_float4(ndcSize.x, ndcSize.y, 0.0f, 0.0f), color };
    vertices[info->vertexCount++] = { ndcPosV4 + simd_make_float4(0.0f, ndcSize.y, 0.0f, 0.0f), color };
    
    vertices[info->vertexCount++] = { ndcPosV4 + simd_make_float4(0.0f, ndcSize.y, 0.0f, 0.0f), color };
    vertices[info->vertexCount++] = { ndcPosV4 + simd_make_float4(0.0f, 0.0f, 0.0f, 0.0f), color };
}

-(void)flushBoxes:(nonnull id<MTLRenderCommandEncoder>)encoder boxRenderInfo:(nonnull struct BoxRenderInfo *)renderInfo
{
    [encoder pushDebugGroup:@"Box Renderer"];
    
    [encoder setRenderPipelineState:mBoundingBoxPipeline];
    
    [encoder setVertexBuffer:renderInfo->vertexBuffer offset:0 atIndex:0];
    
    // Draw the quad.
    [encoder drawPrimitives:MTLPrimitiveTypeLine
                      vertexStart:0
                      vertexCount:renderInfo->vertexCount];
    
    renderInfo->vertexCount = 0;
}

#define MAX_FONT_BUFFER_SIZE (sizeof(struct FontVertex) * 6 * 1000)

-(struct FontRenderInfo)makeFontRenderInfo:(id<MTLDevice>)device
{
    struct FontRenderInfo renderInfo;
    renderInfo.vertexCount = 0;
    renderInfo.vertexBuffer = [device newBufferWithLength:MAX_FONT_BUFFER_SIZE options:MTLResourceStorageModeShared];
    return renderInfo;
}

-(void)pushText:(struct FontRenderInfo *)info text:(char *)text position:(vector_float2)pos viewport:(vector_uint2)viewport
{
    // Pixel width of a character.
    uint32_t pixelCharWidth = 15;
    uint32_t pixelCharHeight = (int)(1.3 * (float)pixelCharWidth);
    
    // Starting position of the text that will rendered.
    vector_float2 pixelCursorPosition = simd_make_float2((float)viewport.x * pos.x, (float)viewport.y * pos.y);
    
    for (uint32_t character = 0; character < strlen(text);)
    {
        char currentCharValue = text[character];
        
        if (currentCharValue == '\n')
        {
            pixelCursorPosition = simd_make_float2((float)viewport.x * pos.x, pixelCursorPosition.y - (float)pixelCharHeight);
            character++;
            continue;
        }
        
        FontCharacter *fontCharData = &mCharacters[(uint32_t)currentCharValue];

        // Top left
        vector_float2 pixelCharacterSize = pixelCharWidth * fontCharData->displaySize;
        vector_float2 pixelCharacterBasePosition = pixelCursorPosition + fontCharData->offset * (float)pixelCharWidth;
        
        vector_float2 ndcBasePosition = pixelCharacterBasePosition;
        ndcBasePosition /= simd_make_float2((float)viewport.x, (float)viewport.y);
        ndcBasePosition *= 2.0f;
        ndcBasePosition -= simd_make_float2(1.0f, 1.0f);
        
        vector_float2 ndcSize = (pixelCharacterSize / simd_make_float2((float)viewport.x, (float)viewport.y)) * 2.0f;
        vector_float2 adjust = simd_make_float2(0.0f, -ndcSize.y);
        
        vector_float2 currentUVs = fontCharData->uvsBase;
        currentUVs.y = 1.0f - currentUVs.y;
        
        // Calclulate the vertices to push for rendering.
        struct FontVertex p0 = { ndcBasePosition + adjust, currentUVs };
        
        currentUVs = fontCharData->uvsBase + simd_make_float2(0.0f, fontCharData->uvsSize.y);
        currentUVs.y = 1.0f - currentUVs.y;
        
        struct FontVertex p1 = { ndcBasePosition + adjust + simd_make_float2(0.0f, ndcSize.y), currentUVs };
        
        currentUVs = fontCharData->uvsBase + simd_make_float2(fontCharData->uvsSize.x, 0.0f);
        currentUVs.y = 1.0f - currentUVs.y;
        
        struct FontVertex p2 = {ndcBasePosition + adjust + simd_make_float2(ndcSize.x, 0.0f), currentUVs};
        
        currentUVs = fontCharData->uvsBase + simd_make_float2(0.0f, fontCharData->uvsSize.y);
        currentUVs.y = 1.0f - currentUVs.y;
        
        struct FontVertex p3 = {ndcBasePosition + adjust + simd_make_float2(0.0f, ndcSize.y), currentUVs};

        currentUVs = fontCharData->uvsBase + simd_make_float2(fontCharData->uvsSize.x, 0.0f);
        currentUVs.y = 1.0f - currentUVs.y;
        
        struct FontVertex p4 = {ndcBasePosition + adjust + simd_make_float2(ndcSize.x, 0.0f), currentUVs};

        currentUVs = fontCharData->uvsBase + fontCharData->uvsSize;
        currentUVs.y = 1.0f - currentUVs.y;
        
        struct FontVertex p5 = {ndcBasePosition + adjust + ndcSize, currentUVs};

        pixelCursorPosition += simd_make_float2((float)pixelCharWidth, 0.0f);

        ++character;
        
        // Push the vertices into the metal vertex buffer
        struct FontVertex *newVertex = (struct FontVertex *)[info->vertexBuffer contents];
        
        newVertex[info->vertexCount++] = p0;
        newVertex[info->vertexCount++] = p1;
        newVertex[info->vertexCount++] = p2;
        newVertex[info->vertexCount++] = p3;
        newVertex[info->vertexCount++] = p4;
        newVertex[info->vertexCount++] = p5;
    }
}

-(void)flushFonts:(id<MTLRenderCommandEncoder>)encoder fontRenderInfo:(nonnull struct FontRenderInfo *)renderInfo
{
    [encoder pushDebugGroup:@"Font Renderer"];
    
    [encoder setRenderPipelineState:mTextPipeline];
    
    [encoder setVertexBuffer:renderInfo->vertexBuffer offset:0 atIndex:0];
    [encoder setFragmentTexture:mFontAtlas atIndex:0];
    
    // Draw the quad.
    [encoder drawPrimitives:MTLPrimitiveTypeTriangle
                      vertexStart:0
                      vertexCount:renderInfo->vertexCount];
    
    renderInfo->vertexCount = 0;
}

@end

#endif
